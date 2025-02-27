import numpy as np

def create_random_infomap(type="default", size=(100,100), source_nr=10, source_size=5):
    matrix = np.zeros(size)
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
    
    match type:
        case "default":
            num_sources = np.random.randint(1, source_nr + 1)
            
            for _ in range(num_sources):
                center_x = np.random.randint(0, size[0])
                center_y = np.random.randint(0, size[1])
                intensity = np.random.uniform(0.5, 1.5)  # Random intensity multiplier
                length_scale = np.random.uniform(1, source_size)  # Random decay factor
                
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                gaussian_field = intensity * np.exp(-distance / length_scale)
                
                matrix += gaussian_field  # Add to the overall information map
        case "point":
            #num_sources = np.random.randint(1, source_nr + 1)
            num_sources = source_nr
            
            for _ in range(num_sources):
                center_x = np.random.randint(0, size[0])
                center_y = np.random.randint(0, size[1])  
                matrix[center_y][center_x] = 1 
    
    return matrix