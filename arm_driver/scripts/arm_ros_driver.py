#!/usr/bin/env python3
import rospy
import serial
import threading
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_srvs.srv import Trigger, SetBool, TriggerResponse
from arm_driver.srv import SetValues, SetValuesResponse
class ArmDriver:
    def __init__(self, port, baudrate, timeout=2):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        rospy.loginfo(f"Serial connection: {self.ser.port} @ {self.ser.baudrate} baud")
        self.lock = threading.Lock()


    def send_command(self, cmd, data=None, binary_mode=False):
        with self.lock:
            try:
                # Бинарный режим
                if binary_mode and data is not None:
                    # Формируем бинарную команду
                    cmd_bytes = cmd.encode('ascii')
                    
                    # Упаковываем данные в бинарный формат
                    data_bytes = b''
                    for value in data:
                        data_bytes += value.to_bytes(4, byteorder='little', signed=True)
                    
                    # Формируем полную команду
                    full_cmd = cmd_bytes + data_bytes + b'\r\n'
                    print(f"[SEND BIN] {full_cmd}")
                    
                    # Отправка
                    self.ser.write(full_cmd)
                    self.ser.flush()
                
                # Текстовый режим
                else:
                    # Формирование текстовой команды
                    if data is not None:
                        cmd_str = f"{cmd}:{','.join(str(x) for x in data)}"
                    else:
                        cmd_str = f"{cmd}:0,0,0,0"
                    
                    full_cmd = cmd_str + "\r\n"
                    print(f"[SEND TXT] {full_cmd.strip()}")
                    self.ser.write(full_cmd.encode('ascii'))
                    self.ser.flush()
                
                # Получение ответа
                rospy.sleep(0.1)
                response = self.ser.readline().decode('utf-8').strip()
                print(f"[RECV] {response}")
                return response
                
            except Exception as e:
                print(f"Error: {str(e)}")
                return None

    def degree_to_ticks(self, degrees):
        return [int(r * 4096 / 360) for r in degrees]

    
    def ticks_to_degree(self, ticks):
        return [int(u * 360 / 4096) for u in ticks]
    
    # ---------- Command Methods ----------
    def enable_motors(self):
        return self.send_command("HRME")
    
    def disable_motors(self):
        return self.send_command("HRMD")
    
    def go_home(self):
        return self.send_command("HRGH")
    
    def set_target_config(self, joints_rad):
        ticks = self.degree_to_ticks(joints_rad)
        return self.send_command("HRSC", ticks, binary_mode=True)
    
    def get_config_abs(self):
        response = self.send_command("HRGA")
        if response and response.startswith("HRGA:"):
            try:
                ticks = [int(x) for x in response[5:].split(',')[:4]]
                return self.ticks_to_degree(ticks)
            except ValueError:
                rospy.logwarn("Invalid HRGA response format")
        return None
    def get_config_relative(self):
        response = self.send_command("HRGC")
        if response and response.startswith("HRGC:"):
            try:
                ticks = [int(x) for x in response[5:].split(',')[:4]]
                return self.ticks_to_degree(ticks)
            except ValueError:
                rospy.logwarn("Invalid HRGC response format")
        return None
    def pnevmo_on(self):
        return self.send_command("HRPI")
    
    def pnevmo_off(self):
        return self.send_command("HRPO")
    
    def conveyer_on(self):
        return self.send_command("HRCO")
    
    def conveyer_off(self):
        return self.send_command("HRCF")

    def get_config_abs(self):
        response = self.send_command("HRGA")
        if response and response.startswith("HRGA:"):
            try:
                ticks = [int(x) for x in response[5:].split(',')[:4]]
                return self.ticks_to_degree(ticks)
            except ValueError:
                return None
        return None
    
    def get_relative_config(self):
        response = self.send_command("HRGC")
        if response and response.startswith("HRGC:"):
            try:
                ticks = [int(x) for x in response[5:].split(',')[:4]]
                return self.ticks_to_degree(ticks)
            except ValueError:
                return None
        return None
    
    def get_target_config(self):
        response = self.send_command("HRGT")
        if response and response.startswith("HRGT:"):
            try:
                ticks = [int(x) for x in response[5:].split(',')[:4]]
                return self.ticks_to_degree(ticks)
            except ValueError:
                return None
        return None
    
    def get_status(self):
        response = self.send_command("HRGS")
        if response and response.startswith("HRGS:"):
            try:
                status_code = int(response[5:].split(',')[0])
                return status_code
            except ValueError:
                return None
        return None
    
    def get_pid_gains(self):
        response = self.send_command("HRGP")
        if response and response.startswith("HRGP:"):
            try:
                gains = [int(x) for x in response[5:].split(',')[:12]]  
                return gains
            except ValueError:
                return None
        return None

class ArmROSDriver:
    def __init__(self):
        rospy.init_node('arm_driver_node')
        
        # Get parameters
        port = rospy.get_param('~port', '/dev/ttyACM0')
        baudrate = rospy.get_param('~baudrate', 115200)
        self.joint_names = rospy.get_param('~joint_names', ['joint1', 'joint2', 'joint3', 'joint4'])
        update_rate = rospy.get_param('~update_rate', 10.0)  # Hz
        
        # Initialize driver
        self.driver = ArmDriver(port, baudrate)
        
        # Services
        rospy.Service('/arm/enable_motors', SetBool, self.handle_enable_motors)
        rospy.Service('/arm/enable_pnevmo', SetBool, self.handle_enable_pnevmo)
        rospy.Service('/arm/go_home', Trigger, self.handle_go_home)
        rospy.Service('/arm/conveyer_on', Trigger, self.handle_conveyer_on)
        rospy.Service('/arm/conveyer_off', Trigger, self.handle_conveyer_off)

        rospy.Service('/arm/set_joints', SetValues, self.handle_set_joints)
        rospy.Service('/arm/set_p_gains', SetValues, self.handle_set_p_gains)
        rospy.Service('/arm/set_i_gains', SetValues, self.handle_set_i_gains)
        rospy.Service('/arm/set_d_gains', SetValues, self.handle_set_d_gains)
        
        rospy.Service('/arm/get_absolute_angles', Trigger, self.handle_get_absolute_angles)
        rospy.Service('/arm/get_relative_angles', Trigger, self.handle_get_relative_angles)
        rospy.Service('/arm/get_target_angles', Trigger, self.handle_get_target_angles)
        rospy.Service('/arm/get_status', Trigger, self.handle_get_status)
        rospy.Service('/arm/get_pid_gains', Trigger, self.handle_get_pid_gains)
        
        rospy.loginfo("Arm driver initialized")
    
    # ---------- Service Handlers ----------
    def handle_enable_motors(self, req):
        if req.data:
            response = self.driver.enable_motors()
            return {'success': bool(response), 'message': response or ''}
        else:
            response = self.driver.disable_motors()
            return {'success': bool(response), 'message': response or ''}
    
    def handle_go_home(self, req):
        response = self.driver.go_home()
        return {'success': bool(response), 'message': response or ''}

    def handle_enable_pnevmo(self, req):
        if req.data:
            response = self.driver.pnevmo_on()
            return {'success': bool(response), 'message': response or ''}
        else:
            response = self.driver.pnevmo_off()
            return {'success': bool(response), 'message': response or ''}

    def handle_conveyer_on(self, req):
        response = self.driver.conveyer_on()
        return {'success': bool(response), 'message': response or ''}
    
    def handle_conveyer_off(self, req):
        response = self.driver.conveyer_off()
        return {'success': bool(response), 'message': response or ''}
    
    def handle_set_joints(self, req):
        if len(req.values) == len(self.joint_names):
            self.driver.set_target_config(req.values)
            return SetValuesResponse(success=True)
        return SetValuesResponse(success=False)
    
    def handle_set_p_gains(self, req):
        if len(req.gains) == 4:
            response = self.driver.set_p_gains(req.gains)
            return SetValuesResponse(success=bool(response), message=response or "")
        return SetValuesResponse(success=False, message="Invalid gains count")
    
    def handle_set_i_gains(self, req):
        if len(req.gains) == 4:
            response = self.driver.set_i_gains(req.gains)
            return SetValuesResponse(success=bool(response), message=response or "")
        return SetValuesResponse(success=False, message="Invalid gains count")
    
    def handle_set_d_gains(self, req):
        if len(req.gains) == 4:
            response = self.driver.set_d_gains(req.gains)
            return SetValuesResponse(success=bool(response), message=response or "")
        return SetValuesResponse(success=False, message="Invalid gains count")

    def handle_get_absolute_angles(self, req):
        angles = self.driver.get_config_abs()
        if angles:
            angles_str = ",".join(f"{a:.4f}" for a in angles)
            return TriggerResponse(success=True, message=angles_str)
        return TriggerResponse(success=False, message="Failed to get absolute angles")
    
    def handle_get_relative_angles(self, req):
        angles = self.driver.get_relative_config()
        if angles:
            angles_str = ",".join(f"{a:.4f}" for a in angles)
            return TriggerResponse(success=True, message=angles_str)
        return TriggerResponse(success=False, message="Failed to get relative angles")
    
    def handle_get_target_angles(self, req):
        angles = self.driver.get_target_config()
        if angles:
            angles_str = ",".join(f"{a:.4f}" for a in angles)
            return TriggerResponse(success=True, message=angles_str)
        return TriggerResponse(success=False, message="Failed to get target angles")
    
    def handle_get_status(self, req):
        status = self.driver.get_status()
        if status is not None:
            return TriggerResponse(success=True, message=str(status))
        return TriggerResponse(success=False, message="Failed to get status")
    
    def handle_get_pid_gains(self, req):
        gains = self.driver.get_pid_gains()
        if gains:
            gains_str = ",".join(str(g) for g in gains)
            return TriggerResponse(success=True, message=gains_str)
        return TriggerResponse(success=False, message="Failed to get PID gains")

if __name__ == '__main__':
    try:
        driver = ArmROSDriver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass