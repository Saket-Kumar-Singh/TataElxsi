from server.models import vehicle

import sys
num1 = 4
num2 = 5
num3 = 6

# for i in range(1, 4):
#     print(eval(f"num{i}")) 

# for i in range(3):
#     exec(f'num{i} = {i}')

# # Check the values
# print(num1, num2)

# print(sys.path)

# vh1 = vehicle(4)
# for i in range(10):
#     print(vh1)

# from datetime import datetime, timedelta
# import time

# # Record the previous time
# prev_time = datetime.now()

# # Wait for 5 seconds
# time.sleep(5)

# # Record the current time after waiting
# curr_time = datetime.now()

# # Calculate the time difference
# time_difference = curr_time - prev_time

# # Print the time difference
# print("Time Difference:", time_difference)
# print("Time Difference:", time_difference.microseconds)
# print("Time Difference:", time_difference)


from datetime import datetime, timedelta
import time

# Record the previous time
prev_time = datetime.now()

# Wait for 5 seconds
time.sleep(5)

# Record the current time after waiting
curr_time = datetime.now()

# Calculate the time difference
time_difference = curr_time - prev_time

# Get the time difference in seconds
time_difference_seconds = time_difference.total_seconds()

# Print the time difference in seconds
print("Time Difference in Seconds:", time_difference_seconds)
