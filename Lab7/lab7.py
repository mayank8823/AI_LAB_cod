# %%
import numpy as np
from skimage import data, io, filters
from matplotlib import pyplot as plt
from PIL import Image
from skimage.transform import resize
import random


cameraman = data.camera()
print("Image Dimensions:", cameraman.shape)
plt.imshow(cameraman, cmap="gray")


def binarize_img(im):
    bin_im = im / 255.0

    bin_im.flags.writeable = True
    bin_im[bin_im < 0.5] = 0
    bin_im[bin_im >= 0.5] = 1
    return bin_im


bin_cameraman = binarize_img(cameraman)

plt.imshow(bin_cameraman, cmap="gray")


def add_noise(img, thresh=0.05):
    N, M = img.shape
    noisy_img = img.copy()
    noise = np.random.rand(N, M)
    noise[noise < 1 - thresh] = 0
    noise[noise >= 1 - thresh] = 1
    noisy_img = (noisy_img + noise) % 2

    return noisy_img


fig = plt.figure(figsize=(12, 6))

for i in range(0, 5 + 1):
    thresh = i / 20

    noisy_img = add_noise(bin_cameraman, thresh)
    plt.subplot(2, 3, i + 1)
    plt.title(f"Noise: {thresh*100:.0f}%")
    plt.axis("off")
    plt.imshow(noisy_img, cmap="gray")

# Show the figure
plt.show()


def get_neighbours(i, j, M, N):
    neighbours = []
    if i > 0:
        neighbours.append([i - 1, j])
    if i < M - 1:
        neighbours.append([i + 1, j])
    if j > 0:
        neighbours.append([i, j - 1])
    if j < N - 1:
        neighbours.append([i, j + 1])

    return neighbours


def enrg(new, old, y, neighbours):
    lmda = -100
    return (new - old) ** 2 + lmda * np.sum(
        (new - y[neighbour[0], neighbour[1]]) ** 2 for neighbour in neighbours
    )


def diff(y, y_old):
    diff = abs(y - y_old) / 2
    return (100.0 * np.sum(diff)) / np.size(y)


def denoise(noisy_img):
    M, N = noisy_img.shape
    y = noisy_img.copy()
    maxiter = 10 * M * N

    for iter in range(maxiter):
        i = np.random.randint(M)
        j = np.random.randint(N)
        neighbours = get_neighbours(i, j, M, N)

        enrg_1 = enrg(1, y[i, j], y, neighbours)
        enrg_0 = enrg(0, y[i, j], y, neighbours)

        if enrg_1 > enrg_0:
            y[i, j] = 1
        else:
            y[i, j] = 0

        if iter % 1000000 == 0:
            print(
                f"Completed {iter} iterations out of {maxiter}. Denoized pixels are: {diff(y, noisy_img)}%"
            )

    return y


fig = plt.figure(figsize=(8, 20))

for i in range(1, 5 + 1):
    thresh = i / 20
    print(f"Denoising for noise level: {thresh*100}")

    noisy_img = add_noise(bin_cameraman, thresh)
    denoised_img = denoise(noisy_img)

    plt.subplot(5, 2, 2 * i - 1)
    plt.title(f"Noise: {thresh*100}%")
    plt.axis("off")
    plt.imshow(noisy_img, cmap="gray")

    plt.subplot(5, 2, 2 * i)
    plt.title(f"Denoised image ({diff(noisy_img, denoised_img)}%)")
    plt.axis("off")
    plt.imshow(denoised_img, cmap="gray")

    print()
plt.show()


# defining numpy arrays and reshaping them into 5x5 matrix
D = np.array(
    [
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
    ]
).reshape(5, 5)
J = np.array(
    [
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        1,
    ]
).reshape(5, 5)
C = np.array(
    [
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        1,
        1,
        1,
        1,
        -1,
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
    ]
).reshape(5, 5)
M = np.array(
    [
        -1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
        -1,
        1,
        1,
        1,
        -1,
    ]
).reshape(5, 5)

# numpy array holding the above four letters
X = np.array([D, J, C, M])

letters = ["D", "J", "C", "M"]

fig = plt.figure(figsize=(6, 6))

# looping through each of the letters in X
for idx, letter in enumerate(X):
    plt.subplot(2, 2, idx + 1)
    plt.title(letters[idx])
    plt.axis("off")
    plt.imshow(letter, cmap="gray")

# ploting
plt.show()


n = X.shape[0]
m = X.shape[1] * X.shape[2]

W = np.zeros((m, m))
for i in range(n):
    W += X[i, :].reshape(-1, 1) @ X[i, :].reshape(1, -1)

np.fill_diagonal(W, 0)

W /= n


def add_error(num_errors=1):
    chosen_letter = random.choice(X)
    letter_with_err = chosen_letter.copy()
    errors = []
    while num_errors:
        i = np.random.randint(5)
        j = np.random.randint(5)

        if (i, j) not in errors:
            errors.append((i, j))
            letter_with_err[i, j] = -letter_with_err[i, j]
            num_errors -= 1

    return chosen_letter, letter_with_err


fig = plt.figure(figsize=(9, 45))

# Iterate over the number of errors to add
for i in range(1, 15 + 1):
    # Add errors to a chosen letter
    chosen_letter, letter_with_err = add_error(i)
    y = letter_with_err.reshape(-1)

    # Initialize error values to enter the loop
    last_erry = i
    erry = 26
    while erry != last_erry:
        last_erry = erry
        yp = np.sign(W @ y)
        erry = np.linalg.norm(yp - y)
        y = yp
    # Plot the original letter
    plt.subplot(15, 3, 3 * (i - 1) + 1)
    plt.title("Without error(s)")
    plt.axis("off")
    plt.imshow(chosen_letter, cmap="gray")
    # Plot the letter with errors
    plt.subplot(15, 3, 3 * (i - 1) + 2)
    plt.title(f"With {i} error(s)")
    plt.axis("off")
    plt.imshow(letter_with_err, cmap="gray")
    # Plot the error-corrected letter
    plt.subplot(15, 3, 3 * (i - 1) + 3)
    plt.title("Error corrected")
    plt.axis("off")
    plt.imshow(y.reshape(5, 5), cmap="gray")

# show figure
plt.show()


# %%
# Number of cities
N = 10

city_x = np.random.rand((10))
city_y = np.random.rand((10))
print("The co-ordinates of the 10 cities are:")
for city in zip(city_x, city_y):
    print(city)

plt.plot(city_x, city_y, "o")
plt.title("Map of cities")

# %%
d = np.zeros((N, N))
# Calculate distance matrix
for i in range(N):
    for j in range(N):
        d[i, j] = np.sqrt((city_x[i] - city_x[j]) ** 2 + (city_y[i] - city_y[j]) ** 2)

A = 500
B = 500
C = 1000
D = 500
alpha = 0.0001


def hopfield():
    u0 = 0.02
    toend = 0
    udao = np.zeros((N, N))
    ctr = 0
    while toend == 0:
        ctr += 1
        # U initialization
        v = np.random.rand(N, N)
        u = np.ones([N, N]) * (-u0 * np.log(N - 1) / 2)

        u += u * 0.91
        for _ in range(1000):
            for vx in range(N):
                for vi in range(N):
                    j1, j2, j3, j4 = 0, 0, 0, 0

                    # derivative 1 (sum over columns j!=vi)
                    for j in range(N):
                        if j != vi:
                            j1 += v[vx, j]
                    j1 *= -A

                    # derivative 2 (sum over rows y!=x)
                    for y in range(N):
                        if y != vx:
                            j2 += v[y, vi]
                    j2 *= -B

                    # derivative 3 (overall sum)
                    j3 = np.sum(v)
                    j3 = -C * (j3 - N)

                    # derivative 4
                    for y in range(N):
                        if y != vx:
                            if vi == 0:
                                j4 += d[vx, y] * (v[y, vi + 1] + v[y, N - 1])
                            elif vi == N - 1:
                                j4 += d[vx, y] * (v[y, vi - 1] + v[y, 0])
                            else:
                                j4 += d[vx, y] * (v[y, vi + 1] + v[y, vi - 1])
                    j4 *= -D
                    udao[vx, vi] = -u[vx, vi] + j1 + j2 + j3 + j4

            # update status and derivatives
            u = u + alpha * udao

            # calculate node value from input potential u
            v = (1 + np.tanh(u / u0)) / 2

            # threshold
            for vx in range(N):
                for vi in range(N):
                    if v[vx, vi] < 0.7:
                        v[vx, vi] = 0
                    if v[vx, vi] >= 0.7:
                        v[vx, vi] = 1

        # testing whether solution is legal
        t1, t2, t3 = 0, 0, 0

        # require total of N-nodes with 1 value
        t1 = 0
        for vx in range(N):
            for vi in range(N):
                t1 += v[vx, vi]

        # allow only one node in each row equal 1
        t2 = 0
        for x in range(N):
            for i in range(N - 1):
                for j in range(i + 1, N):
                    t2 += np.multiply(v[x, i], v[x, j])

        # allow only one node in each column equal 1
        t3 = 0
        for i in range(N):
            for x in range(N - 1):
                for y in range(x + 1, N):
                    t3 += np.multiply(v[x, i], v[y, i])

        # stop the loop after getting the valid solution
        if t1 == N and t2 == 0 and t3 == 0:
            toend = 1
        else:
            toend = 0

    return (v, ctr)


def total_distance(v):
    # Initializing arrays to hold the final city coordinates
    city_x_final = np.zeros((N + 1))
    city_y_final = np.zeros((N + 1))

    # Looping through the cities in the given route to determine their coordinates
    for j in range(N):
        for i in range(N):
            if v[i, j] == 1:
                city_x_final[j] = city_x[i]
                city_y_final[j] = city_y[i]

    # Adding the starting city as the final city to complete the route
    city_x_final[N] = city_x_final[0]
    city_y_final[N] = city_y_final[0]

    # calculate the total distance
    td = 0
    for i in range(N - 1):
        td += np.sqrt(
            (city_x_final[i] - city_x_final[i + 1]) ** 2
            + (city_y_final[i] - city_y_final[i + 1]) ** 2
        )
    td += np.sqrt(
        (city_x_final[N - 1] - city_x_final[0]) ** 2
        + (city_y_final[N - 1] - city_y_final[0]) ** 2
    )

    return (
        td,
        city_x_final,
        city_y_final,
    )  # Returning the total distance and the final city coordinates


v = np.zeros([N, N])

ct = 0

min_dist = 20
best_path = None

# running hoplield network for 20 epochs
for i in range(20):
    v, steps = hopfield()
    td, _, _ = total_distance(v)
    print(f"Epoch {i}: Ran for {steps} steps, total distance {td}")
    if td < min_dist:
        min_dist = td
        best_path = v


def get_route(v):
    route = ""
    for j in range(v.shape[1]):
        route += str(np.argmax(v[:, j])) + " -> "
    return route + str(np.argmax(v[:, 0]))


print(get_route(best_path))


indices = [3, 8, 9, 1, 2, 0, 5, 4, 6, 7, 3]


for i in indices[1:]:
    plt.plot([city_x[i], city_x[i - 1]], [city_y[i], city_y[i - 1]], "-")
plt.show()
