import csv

# Data for the CSV file
data = [
    ["Radar", "Latitude", "Longitude"],
    ["Ashdod", 31.77757586390034, 34.65751251836753],
    ["Kiryat_Gat", 31.602089287486198, 34.74535762921831],
    ["Ofakim", 31.302709659709315, 34.59685294800365],
    ["Tseelim", 31.20184656499955, 34.52669152933695],
    ["Meron", 33.00023023451869, 35.404698698883585],
    ["YABBA", 30.653610411909529, 34.783379139342955],
    ["Modiin", 31.891980958022323, 34.99481765229601],
    ["Gosh_Dan", 32.105913486777084, 34.78624983651992],
    ["Carmel", 32.65365306190331, 35.03028065430696],
]

# Saving the data into a CSV file
filename = "radars2.csv"

with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Data saved to {filename}")
