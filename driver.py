import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def read_csv(file_path):
    data, targets = [], []
    with open(file_path, 'r') as file:
        # process feature names
        line = file.readline().strip()
        features = [f.split("(component")[0].strip() for f in line.split(",")]
        features = np.array(features)

        for line in file:
            row = line.strip().split(',')

             # Convert each element in the row to float
            data.append([float(value) for value in row[:-1]])
            targets.append(float(row[-1]))

    return np.array(data), np.array(targets), features


def normalize(data):
    # min and max vals in each column
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    return (data - min_vals) / (max_vals - min_vals)


def gradient_descent_step(x, y, m, b):
    # predictions
    y_pred = m * x + b

    # errors
    error = y_pred - y

    # gradients
    dm = (2 / len(x)) * np.sum(error * x)
    db = (2 / len(x)) * np.sum(error)

    return dm, db, np.mean(error**2)


def run_experiment(x, y, m, b, alpha=0.1, epsilon=0.01, max_iter=100000):
    for _ in range(max_iter):
        # run gradient descent while difference in parameters is still above threshhold
        dm, db, mse = gradient_descent_step(x, y, m, b)

        if abs(dm) < epsilon and abs(db) < epsilon:
            # if threshold reached on both parameters
            break

        # calculate new parameters
        m -= alpha * dm
        b -= alpha * db

    return 1 - mse/np.var(y), mse, m, b


def evaluate(m, b, x, y):
    # predictions
    y_pred = m * x + b

    # errors
    error = y_pred - y

    mse = np.mean(error**2)
    ve = 1 - mse/np.var(y)

    return ve, mse


def univariate_results(features, x_train, y_train, x_test, y_test, m, b, alpha, epsilon):
    # for each feature, print ve, mse, m, b in univariate model
    for i in range(len(features)-1):
        ve, mse, m, b = run_experiment(x_train[:, i], y_train, m, b, alpha, epsilon, 500000)
        ve_test, mse_test = evaluate(m, b, x_test[:, i], y_test)
        print(features[i])
        print(f"  Training -> MSE = {mse:.3f}, VE = {ve:.3f}, m = {m:.3f}, b = {b:.3f}")
        print(f"  Testing  -> MSE = {mse_test:.3f}, VE = {ve_test:.3f}\n")


def multivariate_gradient_descent(x, y, w, b):
    # predictions
    y_pred = np.dot(x, w) + b

    # errors
    error = y_pred - y

    # gradients
    dw = (2 / len(x)) * np.dot(x.T, error)
    db = (2 / len(x)) * np.sum(error)

    return dw, db, np.mean(error**2)


def run_multi_experiment(x, y, w, b, alpha=0.1, epsilon=0.01, max_iter=100000):
    mse_history = []
    for _ in range(max_iter):
        # run gradient descent while difference in parameters is still above threshhold
        dw, db, mse = multivariate_gradient_descent(x, y, w, b)
        mse_history.append(mse)

        if all(abs(dweights) < epsilon for dweights in dw) and abs(db) < epsilon:
            # if threshold reached on both parameters
            break

        # calculate new parameters
        w -= alpha * dw
        b -= alpha * db

    return 1 - mse/np.var(y), mse, w, b, mse_history


def evaluate_multi(w, b, x, y):
    # predictions
    y_pred = np.dot(x, w) + b

    # errors
    error = y_pred - y

    mse = np.mean(error**2)
    ve = 1 - mse/np.var(y)

    return ve, mse


def plot_mse_loss(mse_history, description, step=1000):
    plt.figure(figsize=(8,5))
    plt.plot(range(0, len(mse_history), step), mse_history[::step], label=description)
    plt.yscale("log")  # log scale for MSE
    plt.xlabel("Iteration")
    plt.ylabel("MSE (log scale)")
    # plt.title("MSE Loss Curve for " + description)
    plt.legend()
    plt.show()


def plot_mse_later_iterations(mse_history, description, start_iter=10000, step=1000):
    plt.figure(figsize=(8,5))
    plt.plot(range(start_iter, len(mse_history), step), mse_history[start_iter::step], label=f"MSE from iteration {start_iter} onward")
    plt.yscale("log")  # log scale for MSE
    plt.xlabel("Iteration")
    plt.ylabel("MSE (log scale)")
    # plt.title("MSE Loss Curve from iteration " + str(start_iter) + " for " + description)
    plt.legend()
    plt.show()


def multivariate_results(x_train, y_train, x_test, y_test, w, b, alpha, epsilon):
    # for each feature, print ve, mse, m, b in univariate model
    ve, mse, w, b, mse_history = run_multi_experiment(x_train, y_train, w, b, alpha, epsilon, 1000000)
    ve_test, mse_test = evaluate_multi(w, b, x_test, y_test)
    print(f"  Training -> MSE = {mse:.3f}, VE = {ve:.3f}, w = {w}, b = {b:.3f}")
    print(f"  Testing  -> MSE = {mse_test:.3f}, VE = {ve_test:.3f}\n")
    return mse_history
    


# --------------------- Setup --------------------- #
DATA, TARGETS, FEATURES = read_csv('Concrete_Data.csv')

X_TEST = DATA[499:629, :]
Y_TEST = TARGETS[499:629]
X_TRAIN = np.concatenate([DATA[:499, :], DATA[629:, :]])
Y_TRAIN = np.concatenate([TARGETS[:499], TARGETS[629:]])

X_TRAIN_NORMALIZED = normalize(X_TRAIN)
X_TEST_NORMALIZED = normalize(X_TEST)

FEATURES = np.delete(FEATURES, len(FEATURES)-1)
FEATURES[-1] = "Compressive Strength"


################################ Part A ################################

# ---------------------- Q1 ---------------------- #

print("****************************************************")
print("Normalized Predictors:\n")
univariate_results(FEATURES, X_TRAIN_NORMALIZED, Y_TRAIN, X_TEST_NORMALIZED, Y_TEST, m=0, b=0, alpha=0.1, epsilon=1e-6)

print("****************************************************")
print("Raw Predictors:\n")
univariate_results(FEATURES, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, m=0, b=0, alpha=1e-6, epsilon=1e-3)


# ------------------- Q2.1, Q2.2 ------------------- #

def q2_1_test():
    x = np.array([[3, 4, 4],
                 [4, 2, 1],
                 [10, 2, 5],
                 [3, 4, 5],
                 [11, 1, 1]])
    y = np.array([3, 2, 8, 4, 5])

    w = np.array([1.0]*x.shape[1])
    b = 1
    alpha = 0.1

    dw, db, _ = multivariate_gradient_descent(x, y, w, b)
    w -= alpha * dw
    b -= alpha * db

    print("Weights: " + str(w))
    print("Bias: " + str(b))

    
q2_1_test()


# -------------------- Q2.3, Q2.4 -------------------- #
print("****************************************************")
print("Normalized Predictors:\n")
mse_history_norm = multivariate_results(X_TRAIN_NORMALIZED, Y_TRAIN, X_TEST_NORMALIZED, Y_TEST, w=np.array([1.0]*X_TRAIN.shape[1]), b=0, alpha=0.1, epsilon=1e-6)

print("****************************************************")
print("Raw Predictors:\n")
mse_history_raw = multivariate_results(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, w=np.array([1.0]*X_TRAIN.shape[1]), b=0, alpha=1e-9, epsilon=1e-3)

plot_mse_loss(mse_history_norm, description="Normalized Predictors")
plot_mse_loss(mse_history_raw, description="Raw Predictors")
plot_mse_later_iterations(mse_history_norm, description="Normalized Predictors")
plot_mse_later_iterations(mse_history_raw, description="Raw Predictors")

################################ Part B ################################

def statsmodels_ols(x_train, y_train, x_test, y_test):
    x_train_const = sm.add_constant(x_train)
    x_test_const = sm.add_constant(x_test)

    model = sm.OLS(y_train, x_train_const).fit()
    print(model.summary())

    # predictions
    y_train_pred = model.predict(x_train_const)
    y_test_pred = model.predict(x_test_const)

    # performance
    mse_train = np.mean((y_train - y_train_pred) **2)
    mse_test = np.mean((y_test - y_test_pred) **2)
    ve_train = 1 - mse_train / np.var(y_train)
    ve_test = 1 - mse_test / np.var(y_test)

    print(f"Training -> MSE = {mse_train:.3f}, VE = {ve_train:.3f}")
    print(f"Testing  -> MSE = {mse_test:.3f}, VE = {ve_test:.3f}")
    print("\nCoefficient and P-values for each feature:")
    print(f"{'Feature':<15}{'Coefficient':>15}{'P-value':>15}")
    max_len = max(len(f) for f in FEATURES.tolist())
    print("-" * (max_len+30))
    for feature, coeff, p in zip(["Intercept"] + FEATURES.tolist(), model.params, model.pvalues):
        sig = " *" if p < 0.05 else ""
        print(f"{feature:<{max_len}}{coeff:>15.3f}{p:>15.2e}{sig}")
    print("\n")

# ---------------------- Q1, Q2 ---------------------- #

print("****************************************************")
print(f"\n--- Raw Predictors ---\n")
raw_model = statsmodels_ols(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)

print(f"--- Normalized Predictors ---\n")
norm_model = statsmodels_ols(X_TRAIN_NORMALIZED, Y_TRAIN, X_TEST_NORMALIZED, Y_TEST)

print(f"--- Log-Transformed Predictors ---\n")
X_TRAIN_LOG = np.log1p(X_TRAIN)
X_TEST_LOG = np.log1p(X_TEST)
log_model = statsmodels_ols(X_TRAIN_LOG, Y_TRAIN, X_TEST_LOG, Y_TEST)



