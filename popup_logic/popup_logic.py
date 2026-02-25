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

        # Initialize first Z-row (the 'front' of the card)
        # Each cell stores the cost of having height h at Z=0, considering the initial configuration
        for h in range(y_max + 1):
            current_col = voxel_grid[x, :, 0]
            # Cost = (number of 0s below h * add_cost) + (number of 1s above h * remove_cost)
            cost = (np.sum(current_col[h:]) * remove_cost) + \
                   ((h - np.sum(current_col[:h])) * add_cost)
            dp[0, h] = cost

        # Fill DP table (Monotonicity: H[z] must be >= H[z+1])
        for z in range(1, z_size):
            for h in range(y_max + 1):
                prev_costs = dp[z-1, :h+1] # Only consider previous heights that are <= current height
                min_prev_h = np.argmin(prev_costs)
                
                current_col = voxel_grid[x, :, z]
                edit_cost = (np.sum(current_col[h:]) * remove_cost) + \
                            ((h - np.sum(current_col[:h])) * add_cost)
                
                dp[z, h] = dp[z-1, min_prev_h] + edit_cost
                parent[z, h] = min_prev_h

        # Backtrack optimal heights
        curr_h = np.argmin(dp[z_size-1, :])
        for z in range(z_size - 1, -1, -1):
            final_grid[x, :curr_h, z] = 1
            curr_h = parent[z, curr_h]

    return final_grid

def generate_blueprint(grid):
    size_x, size_y, size_z = grid.shape
    fig, ax = plt.subplots(figsize=(size_x, size_z + size_y))

    # store bounds for cuts to avoid overlaps
    col_bounds_lower = {}
    col_bounds_upper = {}
    
    for x in range(size_x):
        horizon = 0
        y_prev = 0
        y_min, y_max = 0, 0
        y_min_valley, y_max_valley = None, None
        y_min_mountain, y_max_mountain = None, None
        for z in range(size_z):

            # find top-most voxel at this (x, z)
            column = np.where(grid[x, :, z] == 1)[0]
            y_val = column.max() + 1 if column.size > 0 else 0
            z_val = -size_z + z

            # no fold if there's no change in height
            if y_val <= y_prev:
                continue

            # current line positions
            valley_y = z_val + horizon
            mountain_y = z_val + horizon + y_val
            
            # draw valley folds
            ax.plot([x, x+1], [valley_y, valley_y], color='blue', lw=2)

            # draw mountain folds
            ax.plot([x, x+1], [mountain_y, mountain_y], color='red', lw=2)

            # update baseline and horizon
            horizon += (y_val - y_prev)
            y_prev = y_val

            # update y_min and y_max for this column
            if y_min_valley is None:
                y_min_valley = valley_y
            if y_max_valley is None:
                y_max_valley = valley_y
            if y_min_mountain is None:
                y_min_mountain = mountain_y
            if y_max_mountain is None:
                y_max_mountain = mountain_y
            y_min_valley = min(y_min_valley, valley_y)
            y_max_valley = max(y_max_valley, valley_y)
            y_min_mountain = min(y_min_mountain, mountain_y)
            y_max_mountain = max(y_max_mountain, mountain_y)
        
        # after iterating draw the new horizon line for this column
        # if there were no folds, this will just be a straight line at the current horizon level
        ax.plot([x, x+1], [horizon, horizon], color='blue', lw=2)
        # update y_min and y_max for this column
        if y_min_valley is None:
            y_min_valley = 0
        if y_max_valley is None:
            y_max_valley = 0
        if y_min_mountain is None:
            y_min_mountain = 0
        if y_max_mountain is None:
            y_max_mountain = 0
        y_max_valley = max(y_max_valley, horizon)
        col_bounds_lower[x] = (y_min_valley, y_min_mountain)
        col_bounds_upper[x] = (y_max_valley, y_max_mountain)

    print (col_bounds_lower)
    print (col_bounds_upper)
    
    # add cuts between maximum mountain and valley folds
    for x in range(size_x + 1):
        # get upper and lower bounds of mountain and valley folds for this column and the previous one
        upper_left = col_bounds_upper.get(x-1, (0, 0))
        lower_left = col_bounds_lower.get(x-1, (0, 0))
        upper_right = col_bounds_upper.get(x, (0, 0))
        lower_right = col_bounds_lower.get(x, (0, 0))

        # check whether to draw a cut between mountain folds on the RIGHT
        if upper_left[1] != upper_right[1]:
            # if valley folds are contiguous or within the mountain folds, draw a cut between the mountain folds
            if upper_left[0] == upper_right[0] or (upper_right[0] <= upper_right[1]):
                ax.plot([x, x], [lower_right[1], upper_right[1]], color='black', lw=2)

        # check whether to draw a cut between mountain folds on the LEFT
        if upper_left[1] != upper_right[1]:
            # if valley folds are contiguous or within the mountain folds, draw a cut between the mountain folds
            if upper_left[0] == upper_right[0] or (upper_left[0] <= upper_left[1]):
                ax.plot([x, x], [lower_left[1], upper_left[1]], color='black', lw=2)

        # check whether to draw a cut between valley folds on the RIGHT
        if lower_left[0] != lower_right[0]:
            # if mountain folds are contiguous or within the valley folds, draw a cut between the valley folds
            if lower_left[1] == lower_right[1] or (lower_right[1] >= lower_right[0]):
                ax.plot([x, x], [lower_right[0], upper_right[0]], color='black', lw=2)

        # check whether to draw a cut between valley folds on the LEFT
        if lower_left[0] != lower_right[0]:
            # if mountain folds are contiguous or within the valley folds, draw a cut between the valley folds
            if lower_left[1] == lower_right[1] or (lower_left[1] >= lower_left[0]):
                ax.plot([x, x], [lower_left[0], upper_left[0]], color='black', lw=2)
    
    ax.set_aspect('equal')
    ax.set_xlabel("Width (X)")
    ax.set_ylabel("Front (floor) (-) <--- Center Fold (0) ---> Back (wall) (+)")
    plt.title("Centered Blueprint: Red=Mountain, Blue=Valley, Black=Cut")
    plt.grid(True, which='both', linestyle=':', alpha=0.3)
    plt.show()

def visualize_results(initial, final=None, size=10):
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