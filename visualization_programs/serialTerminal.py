import serial
import time
import serial.tools.list_ports

class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(2048, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)


def choseArduinoSerial():
    baudList = [9600, 19200, 38400, 57600, 115200, 230400, 460800]
    ports = serial.tools.list_ports.comports()
    if len(ports) == 0:
        print("No available port found.")
        exit()
    portList = []
    port = '0'
    i = 0
    print("Serial ports found: ")
    for onePort in ports:
        portList.append(str(onePort))
        print(str(i)+". "+str(onePort))
        i+=1
    val = input("Choose the Arduino Serial Port: ")
    if val == "":
        port = portList[0]
    else:
        port = portList[int(val)]
    porta = port.split(" ")
    port = porta[0]
    print(port + " selected")
    print("Select baudrate: ")
    for baudrate in baudList:
        print(str(i)+". "+str(baudrate))
        i+=1
    print("(Deafaul baudrate: 460800)")
    val = input()
    if val == "":
        baud = 460800
    else:
        baud = baudList[int(val)-1]
    print(f"{baud} baudrate selected")
    
    return port, baud

def SerialRead():
    
    port, baud = choseArduinoSerial()
    old_packet = "0"*44
    values = [0,0,0,0,0,0,0,0,0] #add zero for the angle
    old_values = [0,0,0,0,0,0,0,0,0]
    ser = serial.Serial(port, baud, timeout = None, bytesize=8, parity="N", stopbits=1)
    rl = ReadLine(ser)
    ser.close()
    ser.open()
    start = 0
    new_time = 0
    old_time = 0
    ser.flush()
    time.sleep(0.5)
    ser.write(b"SON")
    try:
        while True:
        # new_time = time.perf_counter_ns()
            
            if ser.in_waiting:
                start = time.perf_counter_ns()
                # Read the transmitted bytes   
                packet = rl.readline()      
                #print(packet_utf)
                #print(round((new_time-old_time)*pow(10,-9),4))
                #old_time = new_time
                try:
                    packet_utf = packet.decode('utf-8', errors='ignore').strip()
                    if len(packet_utf)==36:
                        values[0] = int(packet_utf[0:4])
                        values[1] = int(packet_utf[4:8])
                        values[2] = int(packet_utf[8:12])
                        values[3] = int(packet_utf[12:16])
                        values[4] = int(packet_utf[16:20])
                        values[5] = int(packet_utf[20:24])
                        values[6] = int(packet_utf[24:28])
                        values[7] = int(packet_utf[28:32])
                        values[8] = int(packet_utf[32:])
                        old_values = values
                    #print(received_str)
                except:
                    values = old_values

                end = time.perf_counter_ns()
                #print(f"{round((end-start)*pow(10,-9),4)}")
                #print(round((end-start)*pow(10,-9),4))
                #print(f"Received values: {values}")
                #end = start
                print(f"{values[0]}\t{values[1]}\t{values[2]}\t{values[3]}\t{values[4]}\t{values[5]}\t{values[6]}\t{values[7]}\t{values[8]}")
    except KeyboardInterrupt:
        ser.write(b"SOFF")
        ser.close()
        print("Serial Closed")



if __name__ == "__main__":
    
    try:
        SerialRead()

    except KeyboardInterrupt:
        print("Quitting")
        quit()