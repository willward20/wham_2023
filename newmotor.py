#phase enable robot driver code
from gpiozero import PhaseEnableMotor, LED

pins = [25,13,23] #phase pin, enable pin, LED pin
led = LED(pins[2])
led.on()#set pin 22 high to enable driver

motor = PhaseEnableMotor(pins[0], pins[1])
speed_limit = 10

def drive(speed):
    if speed >= 0:
        motor.forward(speed * speed_limit)
    else:
        motor.backward(speed * speed_limit)
