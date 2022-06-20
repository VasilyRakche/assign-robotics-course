import sys
sys.path.append('../../build')
import numpy as np
import libry as ry
import time
import cv2 as cv

def _segment_redpixels(rgb):
    """
    Compute a binary mask of red pixels
    """
    rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, (  0, 120, 70), ( 10, 255, 255))
    mask2 = cv.inRange(hsv, (170, 120, 70), (180, 255, 255))
    
    mask = mask1 + mask2

    # find contours
    contours = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if len(contours)>0:
        # find largest contour
        largest, idx = 0., None
        for i, c in enumerate(contours):
            if c.shape[0]<10: continue
            if cv.contourArea(c)>largest: 
                largest = cv.contourArea(c)
                idx = i
                 
        if idx is not None:
            # fill contours line in rgb image
            mask = cv.drawContours(np.zeros_like(mask), contours, idx, 255, cv.FILLED)          
            
#             # uncomment the followings to see contours line in rgb image  
#             import matplotlib.pyplot as plt
#             rgb = cv.drawContours(rgb, contours, idx, (0,255,0), 3)
#             plt.figure()
#             plt.imshow(rgb)
#             plt.show()
    
    return mask

def _image_pointcloud(depth, rgb, mask):
    """
    Compute point cloud in pixel coordinate
    """
    mask_pixels = np.where(mask>0)
    pointcloud = np.zeros((mask_pixels[0].shape[0], 3))
    pointcloud[:,0] = mask_pixels[1]  # x pixels
    pointcloud[:,1] = mask_pixels[0]  # y pixels
    pointcloud[:,2] = depth[mask_pixels[0], mask_pixels[1]]
    
    masked_rgb = rgb[mask_pixels]
    return pointcloud, masked_rgb

def _meter_pointcloud(pixel_points, cameraFrame, fxfypxpy):
    """
    Args:
        pixel_points: pointcloud in pixel coordinate
        cameraFrame: rai camera frame
        fxfypxpy: camera intrinsic
    Returns
        points: (N, 3) pointcloud in world frame
        rel_points: (N, 3) pointcloud in camera frame
    """
    x = pixel_points[:,0]
    y = pixel_points[:,1]
    d = pixel_points[:,2]        
    rel_points = np.zeros(np.shape(pixel_points))
    rel_points[:,0] =  d * (x-fxfypxpy[2]) / fxfypxpy[0]
    rel_points[:,1] = -d * (y-fxfypxpy[3]) / fxfypxpy[1]
    rel_points[:,2] = -d
    
    cam_rot = cameraFrame.getRotationMatrix()
    cam_trans = cameraFrame.getPosition()
    points = rel_points @ cam_rot.T + cam_trans
    
    return points, rel_points # (num_points, 3)

def find_redpixels(rgb, depth, cameraFrame, fxfypxpy):
    mask = _segment_redpixels(rgb)
    pixel_points, masked_rgb = _image_pointcloud(depth, rgb, mask)
    obj_points, rel_points = _meter_pointcloud(pixel_points, cameraFrame, fxfypxpy)
    return obj_points, rel_points, masked_rgb


def computeObjectPose(obj_points, obj_size, view=True):
    tmpC = ry.Config()
    tmpC.addFrame("world")

    tmpObj = tmpC.addFrame("object")
    tmpObj.setShape(ry.ST.capsule, obj_size)

    tmpC.makeObjectsFree(["object"]) # To add a joint one can optimize 
    if view: D = tmpC.view()

    for i in range(0, obj_points.shape[0], 10):
        name = "pointCloud" + str(i)
        obj = tmpC.addFrame(name)
        obj.setShape(ry.ST.sphere, [.001])
        obj.setPosition(obj_points[i])
        obj.setColor([1,0,0])

    while True: # distance is minimized strategy 
        tmpObj.setPosition(np.mean(obj_points, axis=0)+0.1*np.random.randn(3))
        quat = np.random.randn(4)
        quat /= (np.linalg.norm(quat)+1e-4)
        tmpObj.setQuaternion(quat)

        komo = tmpC.komo_IK(False)
        komo.clearObjectives()
        komo.addQuaternionNorms()
        for i in range(0, obj_points.shape[0], 10):
            name = "pointCloud" + str(i)
            komo.addObjective([1.], ry.FS.distance, ["object", name], ry.OT.sos, [1e0], [.001]) # Likelihood
        komo.optimize()
        tmpC.setFrameState(komo.getFrameState(0))
        
        if komo.getCosts()<1e-3:
            break
            
    return tmpObj.getPosition(), tmpObj.getQuaternion()

class PickUpCapsule:
    def __init__(self):
        self.tau = 0.01
        self.t = 0
        
        self.setupSim()
        self.setupModel()
        
    def setupSim(self):
        #Let's edit the real world before we create the simulation
        self.RealWorld = ry.Config()
        self.RealWorld.addFile("../../scenarios/pandasTable.g")

        targetObj = self.RealWorld.addFrame("object")
        targetObj.setColor([1.,0,0])
        targetObj.setShape(ry.ST.capsule, [.15, .03])
        targetObj.setPosition([.2, .1, .8])
        targetObj.setQuaternion([1., .1, 0,0])
        targetObj.setMass(1)
        targetObj.setContact(1)

        # instantiate the simulation
        self.S = self.RealWorld.simulation(ry.SimulatorEngine.bullet, True)
        self.S.addSensor("camera")
        self.q0 = self.S.get_q()
        
    def setupModel(self):
        # create your model world
        self.C = ry.Config()
        self.C.addFile('../../scenarios/pandasTable.g')
        self.V = self.C.view()
        
        self.obj_size = self.S.getGroundTruthSize("object") # we assume we knew the object shape&size
        self.obj = self.C.addFrame("object")
        self.obj.setShape(ry.ST.capsule, self.obj_size)
        self.obj.setColor([1,1,0,0.9])
        
        self.cameraFrame = self.C.frame("camera")
        # Intrinsic Params
        camInfo = self.cameraFrame.info()
        f = camInfo['focalLength']
        f = f * camInfo['height']
        self.fxfypxpy = [f, f, camInfo['width']/2, camInfo['height']/2]
        
    def perception(self):
        [rgb, depth] = self.S.getImageAndDepth()
        obj_points, rel_points, masked_rgb = find_redpixels(rgb, depth, self.cameraFrame, self.fxfypxpy)
        if len(obj_points):
            self.cameraFrame.setPointCloud(rel_points, masked_rgb)
            pos, quat = computeObjectPose(obj_points, self.obj_size, view=False)
            self.obj.setPosition(pos)
            self.obj.setQuaternion(quat)
            return True
        else:
            return False

    def wait(self, t=1.):
        for i in range(int(t/self.tau)):
            time.sleep(self.tau)
            self.S.step([], self.tau, ry.ControlMode.none)
            self.t += 1
        
    def executeSpline(self, path, t): #Define so the robot moves with the points         
        self.S.setMoveto(path, t)
        while self.S.getTimeToMove() > 0.:
            time.sleep(self.tau)
            self.S.step([], self.tau, ry.ControlMode.spline)
            self.C.setJointState(self.S.get_q())
            self.t += 1
            
    def openGripper(self):
        print('########################## OPENING ##########################')
        self.S.openGripper("R_gripper", speed=1)
        self.C.attach("world", "object") # Important if we want for example to optimize the pozition of the object (so directly manipulate with the object)
        while self.S.getGripperWidth("R_gripper")<0.05-1e-3:
            time.sleep(self.tau)
            self.S.step([], self.tau, ry.ControlMode.none)
            self.C.setJointState(self.S.get_q())
            self.t += 1
        print('########################## OPENED! ##########################')
        
    def pick(self):
        T = 40
        self.C.setJointState(self.S.get_q())
        
        komo = self.C.komo_path(1.,T,T*self.tau,True)
        komo.addObjective([1.], 
                          ry.FS.positionRel, 
                          ["R_gripperCenter","object"], 
                          ry.OT.eq, 
                          1e2*np.array([[1,0,0],[0,1,0]]));
        
        l = self.obj_size[0]
        komo.addObjective([1.], 
                          ry.FS.positionRel, 
                          ["R_gripperCenter","object"], 
                          ry.OT.ineq, 
                          [0, 0, 1e2],
                          [0.,0.,l/2]);
        komo.addObjective([1.], 
                          ry.FS.positionRel, 
                          ["R_gripperCenter","object"], 
                          ry.OT.ineq, 
                          [0, 0, -1e2],
                          [0.,0.,-l/2]);
        
        komo.addObjective([0.7, 1.], 
                        ry.FS.scalarProductXZ, 
                        ["R_gripperCenter", "object"], 
                        ry.OT.eq, 
                        [1e2],
                        [0.])

        komo.addObjective([0.7, 1.], 
                        ry.FS.scalarProductZZ, 
                        ["R_gripperCenter", "object"], 
                        ry.OT.eq, 
                        [1e2],
                        [0.])

        komo.addObjective([0.7, 1.0], 
                          ry.FS.positionRel, 
                          ["object", "R_gripperCenter"], 
                          ry.OT.eq, 
                          [1e2], 
                          [0,0,-0.1], 
                          order=2)
        
        komo.addObjective([], ry.FS.accumulatedCollisions, 
                          type=ry.OT.eq, scale=[1e1])
        komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1],
                          order=1)
        komo.optimize()
        
        path_q = komo.getPath_qOrg()
        self.executeSpline(path_q, 1.)
        
        
        self.S.closeGripper("R_gripper", speed=1.)
        print('########################## CLOSING ##########################')
        while True:
            time.sleep(self.tau)
            self.S.step([], self.tau, ry.ControlMode.none)
            self.C.setJointState(self.S.get_q())
            self.t += 1
            if self.S.getGripperIsGrasping("R_gripper"): 
                print('########################## CLOSED! ##########################')
                self.C.attach("R_gripper", "object")
                return True
            if self.S.getGripperWidth("R_gripper")<0.001: 
                print('########################## FAILED! ##########################')
                return False
        
            
        
    def lift(self):
        print('########################## LIFTING ##########################')
        T = 30
        self.C.setJointState(self.S.get_q())
        komo = self.C.komo_path(1., T, 1., True)
        target = self.C.getFrame("R_gripperCenter").getPosition()
        target[-1] += 0.5
        target[0] = 0.2
        target[1] = (np.random.rand(1)-.5)/3; 
        komo.addObjective([1.], ry.FS.position, ["R_gripperCenter"], ry.OT.eq, [2e1], target=target)
        komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e1])
        komo.addObjective([1.], ry.FS.qItself, [], ry.OT.eq, [1e1], order=1)
        komo.optimize()
        
        path_q = komo.getPath_qOrg()
        self.
        (path_q, 1.)
            
    def moveToInit(self):
        print('########################## GOtoINIT ##########################')
        self.executeSpline(self.q0, 1.)

    def destroy(self):
        self.S = 0
        self.C = 0
        self.V = 0
        
        
        
if __name__ == '__main__':
    M = PickUpCapsule()
    while True:
        M.openGripper()
        M.wait()
        M.moveToInit()
        success = M.perception()
        if not success:
            print('Perception failed (no red pixels)! Terminating the loop...')
            break
        success = M.pick()
        if success:
            M.lift()
            print('########################### DONE! ###########################')