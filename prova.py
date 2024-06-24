import copy
import time
import train
from argparser import DefaultArguments, get_optimal_parameters
from significance import test_significance

def main():
    # Load the default arguments
    default_args = DefaultArguments()

    # Change if the loss should be printed
    default_args.print_loss = False

    # Change the amount of times the results are averaged here.
    default_args.average_over = 10

    # Change the amount of training steps for each of the datasets here.
    training_steps = {
        "uci_adult": 990,
        "law_school": 990,
        "compas": 470,
        "students-dataset": 200,
    }

    print("The default parameters are:\n", default_args.__dict__)

    # Load the optimal hyperparameters.
    students_params = get_optimal_parameters("students-dataset")
    students_params["train_steps"] = training_steps["students-dataset"]

    # Load the arguments passed to the training function.
    students_args = copy.copy(default_args)
    students_args.dataset = "students-dataset"
    students_args.update(students_params)

    print("Parameters used for the Students dataset:\n", students_params)
    # Start timing.
    students_start = time.time()

    # Train the model.
    train.main(students_args)

    # Save the timing results.
    students_time = (time.time() - students_start) / students_args.average_over
    print(f"Training and evaluating took, on average, {students_time:.0f} seconds per model iteration for Students")

if __name__ == "__main__":
    main()