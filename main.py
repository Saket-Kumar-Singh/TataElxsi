import multiprocessing
import processes

if __name__ == "__main__":
    # Create a multiprocessing Process for the loop process
    loop_process = multiprocessing.Process(target=processes.loop_process)

    # Create a multiprocessing Process for the threading process
    threading_process = multiprocessing.Process(target=processes.threading_process)

    # Start the loop process
    loop_process.start()

    # Start the threading process
    threading_process.start()

    try:
        # Wait for both processes to finish
        loop_process.join()
        threading_process.join()
    except KeyboardInterrupt:
        # If the user interrupts (e.g., Ctrl+C), terminate both processes
        loop_process.terminate()
        threading_process.terminate()
