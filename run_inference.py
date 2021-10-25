from datetime import timedelta, date

import subprocess

if __name__ == "__main__":
    start_date = date(2020, 2, 1)
    end_date = date(2020, 5, 31)
    prev_dt = start_date
    for n in range(7, int((end_date - start_date).days) + 1, 7):
        curr_dt = start_date + timedelta(n)
        curr_dt.strftime("%m/%d/%Y")
        command = f'python main.py --mode=inference --start={prev_dt.strftime("%m/%d/%Y")} --end={curr_dt.strftime("%m/%d/%Y")}'
        print(command)
        subprocess.run(command, shell=True)
        prev_dt = curr_dt
