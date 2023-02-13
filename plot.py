import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.rcParams['font.family'] = "Helvetica"
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams['font.weight'] = "bold"

def plot_depth_error_difference(filename1, filename2, label1, label2, title, x_label, y_label):
    content1 = open(filename1, "r")
    content2 = open(filename2, "r")
    numbers11 = content1.readline().strip(" ").split(" ")
    numbers22 = content2.readline().strip(" ").split(" ")
    numbers33 = open("sift_translation_error", "r").readline().strip(" ").split(" ")
    numbers44 = open("surf_translation_error", "r").readline().strip(" ").split(" ")
    numbers55 = open("shitomasi_translation_error", "r").readline().strip(" ").split(" ")
    numbers66 = open("fast_translation_error", "r").readline().strip(" ").split(" ")
    #print(numbers1)
    numbers1, numbers2, numbers3, numbers4, numbers5, numbers6 = [], [], [], [], [], []
    a, b, c, d, e, f = 0, 0, 0, 0, 0, 0
    ab_1, ab_2, ab_3, ab_4, ab_5, ab_6 = 0, 0, 0, 0, 0, 0
    for i in range(0, len(numbers11)):
        if numbers11[i] == "nan" or numbers22[i] == "nan" or numbers33[i] == "nan" or numbers44[i] == "nan":
            continue
        if numbers55[i] == "nan" or numbers66[i] == "nan":
            continue
        if float(numbers11[i]) >= 0.1:
            ab_1 = ab_1 + 1
        if float(numbers22[i]) >= 0.1:
            ab_2 = ab_2 + 1
        if float(numbers33[i]) >= 0.1:
            ab_3 = ab_3 + 1
        if float(numbers44[i]) >= 0.1:
            ab_4 = ab_4 + 1
        if float(numbers55[i]) >= 0.1:
            ab_5 = ab_5 + 1
        if float(numbers66[i]) >= 0.1:
            ab_6 = ab_6 + 1
        a = a + float(numbers11[i])
        b = b + float(numbers22[i])
        c = c + float(numbers33[i])
        d = d + float(numbers44[i])
        e = e + float(numbers55[i])
        f = f + float(numbers66[i])
        numbers1.append(float(numbers11[i]))
        numbers2.append(float(numbers22[i]))
        numbers3.append(float(numbers33[i]))
        numbers4.append(float(numbers44[i]))
        numbers5.append(float(numbers55[i]))
        numbers6.append(float(numbers66[i]))
    #print(len(numbers1))
    #print(numbers1)
    x = [i for i in range(0, len(numbers1))]
    fig, ax = plt.subplots()
    ax.plot(x, numbers1, label=label1, color="red")
    ax.plot(x, numbers2, label=label2, color="blue")
    ax.plot(x, numbers3, label="SIFT", color="green")
    ax.plot(x, numbers4, label="SURF", color="orange")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.fill_between(x, numbers1, facecolor="red", alpha=0.15)
    plt.fill_between(x, numbers2, facecolor="blue", alpha=0.15)
    plt.fill_between(x, numbers3, facecolor="green", alpha=0.15)
    plt.fill_between(x, numbers4, facecolor="orange", alpha=0.15)
    ax.legend()
    ax.set_title(title)
    plt.show()
    print("brisk: ", round(a/len(x), 4), "orb", round(b/len(x), 4), "sift", round(c/len(x), 4), "surf", round(d/len(x), 4), "shitomasi", round(e/len(x), 4), "fast", round(f/len(x), 4))
    print("brisk: ", ab_1, "orb", ab_2, "sift", ab_3, "surf", ab_4, "shitomasi", ab_5, "fast", ab_6)
    print(numbers1.index(min(numbers1)))
    print(numbers2.index(min(numbers2)))
    print(numbers3.index(min(numbers3)))
    print(numbers4.index(min(numbers4)))
    print(numbers5.index(min(numbers5)))
    print(numbers6.index(min(numbers6)))
    

def plot_depth_error_differences(filename1, filename2, label1, label2, title, x_label, y_label):
    content1 = open(filename1, "r")
    content2 = open(filename2, "r")
    numbers11 = content1.readline().strip(" ").split(" ")
    numbers22 = content2.readline().strip(" ").split(" ")
    numbers33 = open("7point_translation_error", "r").readline().strip(" ").split(" ")
    numbers44 = open("8point_translation_error", "r").readline().strip(" ").split(" ")
    #print(numbers1)
    numbers1, numbers2, numbers3, numbers4 = [], [], [], []
    a, b, c, d = 0, 0, 0, 0
    ab_1, ab_2, ab_3, ab_4 = 0, 0, 0, 0
    for i in range(0, len(numbers11)):
        if numbers11[i] == "nan" or numbers22[i] == "nan" or numbers33[i] == "nan" or numbers44[i] == "nan":
            continue
        if float(numbers11[i]) >= 0.1:
            ab_1 = ab_1 + 1
        if float(numbers22[i]) >= 0.1:
            ab_2 = ab_2 + 1
        if float(numbers33[i]) >= 0.1:
            ab_3 = ab_3 + 1
        if float(numbers44[i]) >= 0.1:
            ab_4 = ab_4 + 1
        a = a + float(numbers11[i])
        b = b + float(numbers22[i])
        c = c + float(numbers33[i])
        d = d + float(numbers44[i])
        numbers1.append(float(numbers11[i]))
        numbers2.append(float(numbers22[i]))
        numbers3.append(float(numbers33[i]))
        numbers4.append(float(numbers44[i]))
    #print(len(numbers1))
    #print(numbers1)
    x = [i for i in range(0, len(numbers1))]
    fig, ax = plt.subplots()
    ax.plot(x, numbers1, label=label1, color="red")
    ax.plot(x, numbers2, label=label2, color="blue")
    ax.plot(x, numbers3, label="7Point", color="green")
    ax.plot(x, numbers4, label="8Point", color="orange")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.fill_between(x, numbers1, facecolor="red", alpha=0.15)
    plt.fill_between(x, numbers2, facecolor="blue", alpha=0.15)
    plt.fill_between(x, numbers3, facecolor="green", alpha=0.15)
    plt.fill_between(x, numbers4, facecolor="orange", alpha=0.15)
    ax.legend()
    ax.set_title(title)
    plt.show()
    print("ransac ", round(a/len(x), 4), "lmeds", round(b/len(x), 4), "7point", round(c/len(x), 4), "8point", round(d/len(x), 4))
    print("ransac ", ab_1, "lmeds", ab_2, "7point", ab_3, "8point", ab_4)
    print(numbers1.index(min(numbers1)))
    print(numbers2.index(min(numbers2)))
    print(numbers3.index(min(numbers3)))
    print(numbers4.index(min(numbers4)))
    
def plot_depth_error_differenceo(filename1, filename2, label1, label2, title, x_label, y_label):
    content1 = open(filename1, "r")
    content2 = open(filename2, "r")
    numbers11 = content1.readline().strip(" ").split(" ")
    numbers22 = content2.readline().strip(" ").split(" ")
    #print(numbers1)
    numbers1, numbers2 = [], []
    for i in range(0, len(numbers11)):
        if numbers11[i] == "nan" or numbers22[i] == "nan":
            continue
        numbers1.append(float(numbers11[i]))
        numbers2.append(float(numbers22[i]))
    #print(len(numbers1))
    #print(numbers1)
    x = [i for i in range(0, len(numbers1))]
    fig, ax = plt.subplots()
    ax.plot(x, numbers1, label=label1, color="red")
    ax.plot(x, numbers2, label=label2, color="blue")
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    plt.fill_between(x, numbers1, facecolor="red", alpha=0.15)
    plt.fill_between(x, numbers2, facecolor="blue", alpha=0.15)
    ax.legend()
    ax.set_title(title)
    plt.show()
    print(min(numbers1), numbers1.index(min(numbers1)))
    print(min(numbers2), numbers2.index(min(numbers2)))
    
plot_depth_error_difference("brisk_translation", 
                            "orb_translation_error",
                            "Block Matching",
                            "Semi-global Block Matching",
                            "Depth Error",
                            "Different image pairs",
                            "Average depth map error")