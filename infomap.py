import numpy as np

def create_random_infomap(type="default", size=(100,100), source_nr=10, source_size=5, random=False):
    matrix = np.zeros((100,100))
    x, y = np.meshgrid(np.arange(100), np.arange(100), indexing='ij')
    #matrix = np.zeros(size)
    # x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
    matrix[0:100,0:50-int(size[1]/2)] = np.nan
    matrix[0:100,50+int(size[1]/2):100] = np.nan
    matrix[50+int(size[0]/2):100,0:100] = np.nan
    matrix[0:50-int(size[0]/2),0:100] = np.nan
    match type:
        case "default":
            if random:
                num_sources = np.random.randint(1, source_nr + 1)
            else:
                num_sources = source_nr
            
            for _ in range(num_sources):
                center_x = np.random.randint(50-int(size[0]/2), 50+int(size[0]/2))
                center_y = np.random.randint(50-int(size[1]/2), 50+int(size[1]/2))
                if random:
                    intensity = np.random.uniform(0.5, 1.5)  # Random intensity multiplier
                    length_scale = np.random.uniform(1, source_size)  # Random decay factor
                else:
                    intensity = 1
                    length_scale = source_size
                
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                gaussian_field = intensity * np.exp(-distance / length_scale)
                
                matrix += gaussian_field  # Add to the overall information map
        case "point":
            #num_sources = np.random.randint(1, source_nr + 1)
            num_sources = source_nr
            
            for _ in range(num_sources):
                center_x = np.random.randint(50-int(size[0]/2), 50+int(size[0]/2))
                center_y = np.random.randint(50-int(size[1]/2), 50+int(size[1]/2))
                matrix[center_y][center_x] = 1 
        case "fixed":
            for center_x in range(0,100,20):
                for center_y in range(0,100,15):
                    matrix[center_y][center_x] = 1 
    
    return matrix