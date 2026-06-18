import RPi.GPIO as GPIO
import time
GPIO.setwarnings(False)
from hx711 import HX711

hx = HX711(dout_pin=20, pd_sck_pin=21)
hx.reset()

print("Getting zero offset... remove everything from scale")
time.sleep(2)

# Get offset (empty scale reading)
zero_readings = hx.get_raw_data(times=15)
offset = sum(zero_readings) / len(zero_readings)
print(f"Zero offset: {offset:.1f}")

known_weight = float(input("\nEnter weight of your object in grams (e.g. 500): "))
print(f"Place your {known_weight}g object on scale then press Enter")
input()

time.sleep(0.5)
weight_readings = hx.get_raw_data(times=15)
avg = sum(weight_readings) / len(weight_readings)

factor = (avg - offset) / known_weight

print(f"\n--- Results ---")
print(f"Zero offset  : {offset:.1f}")
print(f"With weight  : {avg:.1f}")
print(f"Difference   : {avg - offset:.1f}")
print(f"\nCALIB_FACTOR = {factor:.2f}")
print(f"\nCopy this number into main.py → CALIB_FACTOR = {factor:.2f}")

GPIO.cleanup()
