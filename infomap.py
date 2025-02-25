import numpy as np

def create_random_infomap(size=(100,100), range_nr=5, range_size=10):
    matrix = np.zeros(size)
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
    
    num_sources = np.random.randint(1, range_nr + 1)
    
    for _ in range(num_sources):
        center_x = np.random.randint(0, size[0])
        center_y = np.random.randint(0, size[1])
        intensity = np.random.uniform(0.5, 1.5)  # Random intensity multiplier
        length_scale = np.random.uniform(1, range_size)  # Random decay factor
        
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        gaussian_field = intensity * np.exp(-distance / length_scale)
        
        matrix += gaussian_field  # Add to the overall information map
    
    return matrix