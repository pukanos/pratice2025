import serial
import time

try:
    ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=2)
    ser.close()
    ser.open()
    # print("Serial connection established.")
    print(f'self.serial.open() DONE desc {ser.port} brate {ser.baudrate}')
    serial_commands = {'get_config_relative':'HRGC','get_target_config':'HRGT','get_config_abs':'HRGA', 'get_error':'HRGE', 'get_xyz':'HRGO', 'get_zero_config':'HRGZ', 'get_wifi_state':'HRGW',
            'set_target_config':'HRSC', 'set_zero_config':'HRSZ','echo_enable':'HRO','error_command':'HREC', 'disable_motors':'HRMD', 'enable_motors':'HRME', 'get_pid_gains':'HRGP', 'set_pid_gains':'HRSP',
            'go_zero_config':'HRGH', 'position_control_on':'HRSP', 'position_control_off':'HRSV', 'fatal_error': 'HRFE', 'get_status': 'HRGS',
            'pnevmo_on':'HRPI','pnevmo_off':'HRPO', 'conveyer_on':'HRCO', 'conveyer_off':'HRCF'}
    while True:
        cmd = input("cmd input ") # отлов плохих ?
        # cmd = serial_commands['get_config_relative']
        ser.write(cmd.encode(encoding = 'ascii', errors = 'strict'))
        out = b''
        out += ser.readline()
        if out != '':
        # if len(out) != 0:
            # print('a')
            # print(out.decode("utf-8"))
            decoded_text = out.decode("utf-8")
            print(f">> Received String length({len(out)}): {decoded_text}")
            time.sleep(1)
                
        
except serial.SerialException as se:
    print("Serial port error:", str(se))

except KeyboardInterrupt:
    pass

finally:
    # Close the serial connection
    if ser.is_open:
        ser.close()
        print("Serial connection closed.")