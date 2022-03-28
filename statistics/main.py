import json
import matplotlib.pyplot as plt

def scene_boundary_proportion(data):
    probs = []
    for movie in data.values():
        scene_list = list(movie.values())
        n_scene = len(scene_list)
        last_scene = scene_list[-1]
        last_scene_shots = last_scene['shot']
        n_shot = int(last_scene_shots[len(last_scene_shots)-1])

        p = int((n_scene / (n_shot-1)) * 100)
        probs.append(p)

    plt.hist(probs)
    plt.xlabel('proportion')
    plt.ylabel('# movies')
    plt.show()

def avg_scene_length(data):
    d = []
    for movie in data.values():
        scene_list = movie.values()
        for scene in scene_list:
            scene_len = len(scene['shot'])
            d.append(scene_len)

    plt.hist(d, bins=range(1, 50))
    plt.xlabel('scene length')
    plt.ylabel('# scene')
    plt.show()

def avg_scene_length_per_movie(data):
    y = []
    for movie in data.values():
        sum = 0
        scene_list = movie.values()
        for scene in scene_list:
            scene_len = len(scene['shot'])
            sum += scene_len
        y.append(sum / len(scene_list))
    x = [i for i in range(0, len(y))]

    plt.scatter(x, y, s=3)
    plt.xlabel('movie')
    plt.ylabel('avg scene length')
    plt.show()

def visualize_per_movie(data):
    for id, movie in data.items():
        d = []
        scene_list = movie.values()
        for scene in scene_list:
            scene_len = len(scene['shot'])
            d.append(scene_len)

        plt.plot(d)
        plt.title(id)
        plt.xlabel('time')
        plt.ylabel('scene length')
        plt.show()

def main():
    with open('scene_movie318.json', 'r') as f:
        data = json.load(f)

    #scene_boundary_proportion(data)
    #avg_scene_length(data)
    #avg_scene_length_per_movie(data)
    visualize_per_movie(data)

if __name__ == '__main__':
    main()