import numpy as np
import matplotlib.pyplot as plt

def optimize_popup_volume(voxel_grid, add_cost=1.0, remove_cost=1.0):
    """
    Optimizes a 3D voxel grid with custom edit costs.
    add_cost: Weight for adding a missing support voxel.
    remove_cost: Weight for removing a floating or non-monotonic voxel.
    """
    x_size, y_max, z_size = voxel_grid.shape
    final_grid = np.zeros_like(voxel_grid)

    for x in range(x_size):
        dp = np.full((z_size, y_max + 1), float('inf'))
        parent = np.zeros((z_size, y_max + 1), dtype=int)

        # z=0 is the FRONT of the card
        # Each cell stores the cost of having height h at Z=0, considering the initial configuration
        for h in range(y_max + 1):
            current_col = voxel_grid[x, :, 0]
            # Cost = (number of 0s below h * add_cost) + (number of 1s above h * remove_cost)
            cost = (np.sum(current_col[h:]) * remove_cost) + \
                   ((h - np.sum(current_col[:h])) * add_cost)
            dp[0, h] = cost

        # complete DP table (monotonic e.g. H[z] must be >= H[z+1])
        for z in range(1, z_size):
            for h in range(y_max + 1):
                prev_costs = dp[z-1, :h+1] # only use previous heights that are <= current height
                min_prev_h = np.argmin(prev_costs)
                
                current_col = voxel_grid[x, :, z]
                edit_cost = (np.sum(current_col[h:]) * remove_cost) + \
                            ((h - np.sum(current_col[:h])) * add_cost)
                
                dp[z, h] = dp[z-1, min_prev_h] + edit_cost
                parent[z, h] = min_prev_h

        # backtrack via parent pointers
        # fill height by placing ones in the final grid up to the chosen height for each z
        curr_h = np.argmin(dp[z_size-1, :])
        for z in range(z_size - 1, -1, -1):
            final_grid[x, :curr_h, z] = 1
            curr_h = parent[z, curr_h]

    return final_grid

def generate_blueprint(grid):
    size_x, size_y, size_z = grid.shape
    fig, ax = plt.subplots(figsize=(size_x, (size_z + size_y) / 2))

    # Map to store all horizontal Y-levels for each column x
    # Format: { x_index: {y_level_1, y_level_2, ...} }
    column_folds = {}

    for x in range(size_x):
        horizon = 0
        y_prev = 0
        levels = [] # Every strip starts with the fold at 0
        
        for z in range(size_z):
            column = np.where(grid[x, :, z] == 1)[0]
            y_val = column.max() + 1 if column.size > 0 else 0
            z_val = -size_z + z

            if y_val <= y_prev:
                continue

            v_y = z_val + horizon
            m_y = z_val + horizon + (y_val - y_prev)
            
            ax.plot([x, x+1], [v_y, v_y], color='blue', lw=2)
            ax.plot([x, x+1], [m_y, m_y], color='red', lw=2)

            levels.append(v_y)
            levels.append(m_y)
            horizon += (y_val - y_prev)
            y_prev = y_val
        
        ax.plot([x, x+1], [horizon, horizon], color='blue', lw=2)
        levels.append(horizon)
        column_folds[x] = levels

    # draw cuts
    for x in range(1, size_x):
        left_levels = column_folds.get(x-1, [])
        right_levels = column_folds.get(x, [])
        all_levels = left_levels + right_levels
        
        # get only levels appearing only once at this boundary
        unique_levels = [level for level in all_levels if all_levels.count(level) == 1]
        unique_levels.sort()

        if len(unique_levels) < 2:
            continue

        ax.plot([x, x], [unique_levels[0], unique_levels[-1]], color='black', lw=2)

    ax.set_aspect('equal')
    ax.axhline(0, color='grey', alpha=0.3)
    plt.title("Smart Blueprint: Cuts truncated by continuous folds")
    plt.show()

def visualize_results(initial, final=None, size=10):
    """
    Visualizes the initial and final voxel grids in 3D.
    
    initial: 3D numpy array of shape (size, size, size) representing the initial voxel configuration.
    final: Optional 3D numpy array of the same shape representing the optimized voxel configuration.
    size: The size of the voxel grid (assumed cubic).
    """
    plt.close('all') 
    fig = plt.figure(figsize=(10, 5) if final is not None else (6, 6), layout='constrained')
    
    def setup_ax(ax, data, title, color):
        ax.voxels(data, edgecolor='k', facecolors=color, alpha=0.8)
        ax.set_title(title)
        
        # --- STRICT POSITIVE ORIGIN ---
        # This ensures the plot always starts at 0 and goes to 'size'
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_zlim(0, size)
        
        # --- PHYSICAL CARD LABELS ---
        # X stays as Width
        ax.set_xlabel('X (<- Left | Right ->)')
        
        # Y is the height relative to the floor
        ax.set_ylabel('Y (^ Height)')
        
        # Z is the depth coming away from the back wall
        ax.set_zlabel('Z (<- Back Wall | Front ->)')
        
        # Optional: Add text annotations for the "Planes"
        # These put labels directly on the "paper" surfaces
        # ax.text(size/2, size, 0, "BACK WALL", color='red', fontweight='bold')
        # ax.text(size/2, 0, size/2, "FLOOR", color='blue', fontweight='bold')
        
        # Visual Polish: Make the "Paper" planes visible
        ax.xaxis.set_pane_color((0.95, 0.95, 0.95, 1.0)) # Light grey floor
        ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))    # Slightly darker back wall
        
        # Optional: Add grid ticks for every voxel
        ax.set_xticks(range(size + 1))
        ax.set_yticks(range(size + 1))
        ax.set_zticks(range(size + 1))
        
        # Ensure the proportions are equal (1 unit X = 1 unit Y)
        ax.set_box_aspect([1, 1, 1]) 
        ax.view_init(elev=20, azim=45, roll=70)

    if final is None:
        ax = fig.add_subplot(111, projection='3d')
        setup_ax(ax, initial, "Voxel Grid", 'orange')
    else:
        ax1 = fig.add_subplot(121, projection='3d')
        setup_ax(ax1, initial, "Initial (Invalid)", 'orange')
        
        ax2 = fig.add_subplot(122, projection='3d')
        setup_ax(ax2, final, "Optimized (Valid)", 'teal')
    
    plt.show()