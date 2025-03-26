# Intersection over union

def IoU(Target, Prediction):
    Union = 0
    Intersection = 0
    for i in range(len(Target)):
        Union += ((Target[i] == 1) | (Prediction[i] == 1)).sum()
        Intersection += ((Target[i] == 1) & (Prediction[i] == 1)).sum()
    return Intersection/Union