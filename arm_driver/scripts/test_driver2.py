import serial
import time
import struct
import math

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

def test():
    # Тестовая функция для проверки всех команд
    driver = ArmDriver(port='/dev/ttyACM0')
    
    # Проверка статуса
    status = driver.get_status()
    print(f"Status: {status}")
    # # Включение моторов
    driver.enable_motors()
    time.sleep(1)
    print(f"Absolute position: {driver.get_config_abs()}")
    print(f"Relative position: {driver.get_config_relative()}")

    driver.pnevmo_on()
    time.sleep(2)
    driver.pnevmo_off()
    time.sleep(1)
    # # Движение в целевую позицию
    target_rad = [0,0,-40,0]
    # target_rad = [90,30,-90,0]
    target_units = driver.radians_to_units(target_rad)
    print(f"Target units: {target_units}")
    driver.set_target_config(target_units)
    time.sleep(10)
    print(f"Absolute position: {driver.get_config_abs()}")
    
    time.sleep(2)
    
    # Возврат в домашнюю позицию
    driver.go_zero_config()
    time.sleep(2)
    
    # Управление пневматикой
    driver.pnevmo_on()
    time.sleep(1)
    driver.pnevmo_off()
    
    # Управление конвейером
    driver.conveyer_on()
    time.sleep(2)
    driver.conveyer_off()
    
    # Отключение моторов
    driver.disable_motors()
    
    # Проверка дополнительных команд
    print(f"Zero config: {driver.get_zero_config()}")
    print(f"Error state: {driver.get_error()}")
    print(f"Target position: {driver.get_target_config()}")
    print(f"P gains: {driver.get_p_gains()}")
    print(f"I gains: {driver.get_i_gains()}")
    print(f"D gains: {driver.get_d_gains()}")


if __name__ == '__main__':
    test()