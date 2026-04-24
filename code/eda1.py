import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_event_frames(filepath, dt_ms=30):
    """
    events.txt loaded, according to format in the file the data has been read and saved. Aligning timesteps to rgb frame duration.
    """
    print(f"Reading just enough data for two {dt_ms}ms frames...")
    
    df = pd.read_csv(
        filepath, 
        sep=r'\s+', 
        comment='#', 
        names=['timestamp', 'x', 'y', 'polarity'],
        dtype={'x': np.uint16, 'y': np.uint16, 'polarity': np.int8},
        nrows=300000 
    )
    
    # Synchronize timestamps to t=0s for aligning with image data
    t0 = df['timestamp'].iloc[0]
    df['t_sec'] = df['timestamp'] - t0
    
    dt_sec = dt_ms / 1000.0
    
    # Asynchronize events to synchronized time steps = 30ms
    events_frame1 = df[(df['t_sec'] >= 0.0) & (df['t_sec'] < dt_sec)]
    events_frame2 = df[(df['t_sec'] >= dt_sec) & (df['t_sec'] < (2 * dt_sec))]
    
    print(f"Events in Frame 1: {len(events_frame1):,}")
    print(f"Events in Frame 2: {len(events_frame2):,}")
    
    # Blank Images initialization to append events data based on polarity
    img1 = np.zeros((180, 240), dtype=np.float32)
    img2 = np.zeros((180, 240), dtype=np.float32)
    
    def accumulate_events(img, frame_df):
        """Vectorized event accumulation without loops."""
        if frame_df.empty:
            return img
            
        # Extract coordinates as pure integers
        x = frame_df['x'].to_numpy(dtype=int)
        y = frame_df['y'].to_numpy(dtype=int)
        
        # Map polarity: if your dataset uses 0 for OFF and 1 for ON
        # we convert 0s to -1s for visual contrast
        p = frame_df['polarity'].to_numpy(dtype=np.float32)
        p = np.where(p == 0, -1.0, 1.0) 
        
        # np.add.at safely adds multiple events happening at the same pixel
        np.add.at(img, (y, x), p)
        return img

    # Accumulate the events instantly
    img1 = accumulate_events(img1, events_frame1)
    img2 = accumulate_events(img2, events_frame2)

    return img1, img2

# --- Execution & Visualization ---
# Replace with your actual path
file_path = "/kaggle/input/datasets/gogo827jz/davis-240c-datasets/boxes_6dof/boxes_6dof/events.txt"

# Generate the two frames
frame1, frame2 = generate_event_frames(file_path, dt_ms=30)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].imshow(frame1, cmap='gray', vmin=-3, vmax=3)
axes[0].set_title('Frame 1 (0 to 30ms)')
axes[0].axis('off')

im2 = axes[1].imshow(frame2, cmap='gray', vmin=-3, vmax=3)
axes[1].set_title('Frame 2 (30ms to 60ms)')
axes[1].axis('off')

plt.tight_layout()
plt.show()