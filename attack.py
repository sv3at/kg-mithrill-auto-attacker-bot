import pyautogui
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import time
import csv
import os
from datetime import datetime
import argparse

# Set PyAutoGUI safety features
pyautogui.PAUSE = 0.02  # Further reduced pause for speed
pyautogui.FAILSAFE = True  # Move mouse to corner to abort

#python attack.py --username John --power 484Fire --searches 20 --delay 3

# Statistics tracking
stats = {
    'start_time': '',
    'username': '',
    'power': '',
    'end_time': '',
    'total_attacks': 0,
    'successful_attacks': 0,
    'failed_attacks': 0,
    'searches_performed': 0,
    'troops_returned': 0
}

def save_statistics_to_csv(username, power):
    """
    Save statistics to CSV file with username and timestamp.
    """
    csv_file = 'mine_attack_statistics.csv'
    file_exists = os.path.isfile(csv_file)
    
    # Prepare data
    stats['username'] = username
    stats['power'] = power
    stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate success rate
    if stats['total_attacks'] > 0:
        success_rate = stats['successful_attacks'] / stats['total_attacks'] * 100
    else:
        success_rate = 0.0
    
    # Write to CSV
    with open(csv_file, 'a', newline='') as f:
        fieldnames = ['username', 'power', 'date_time', 'start_time', 'end_time', 'total_attacks', 
                     'successful_attacks', 'failed_attacks', 'success_rate', 
                     'searches_performed', 'troops_returned']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data
        writer.writerow({
            'username': stats['username'],
            'power': stats['power'],
            'date_time': stats['end_time'],
            'start_time': stats['start_time'],
            'end_time': stats['end_time'],
            'total_attacks': stats['total_attacks'],
            'successful_attacks': stats['successful_attacks'],
            'failed_attacks': stats['failed_attacks'],
            'success_rate': f"{success_rate:.1f}%",
            'searches_performed': stats['searches_performed'],
            'troops_returned': stats['troops_returned']
        })
    
    print(f"\n✓ Statistics saved to {csv_file}")


def find_game_window():
    """
    Detect the LDPlayer game window boundaries.
    """
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    
    # Convert to HSV to detect the yellow/tan game background
    hsv = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2HSV)
    
    # Detect the tan/yellow game area
    lower_tan = np.array([15, 50, 100])
    upper_tan = np.array([40, 200, 255])
    
    mask = cv2.inRange(hsv, lower_tan, upper_tan)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (should be the game area)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h), screenshot_cv
    
    return None, screenshot_cv


def get_mine_area(game_bounds):
    """
    Calculate the central mine area, excluding edges where plants/trees are.
    Returns the cropped mine area boundaries.
    """
    x, y, w, h = game_bounds
    
    # Exclude outer edges
    margin_x = int(w * 0.10)
    margin_y_top = int(h * 0.15)
    margin_y_bottom = int(h * 0.20)
    
    mine_x = x + margin_x
    mine_y = y + margin_y_top
    mine_w = w - (2 * margin_x)
    mine_h = h - margin_y_top - margin_y_bottom
    
    return (mine_x, mine_y, mine_w, mine_h)


def adjust_mine_click_position(red_mark_position, offset_y=30):
    """
    Adjust the click position from red X mark to the actual mine structure below it.
    """
    x, y = red_mark_position
    adjusted_y = y + offset_y
    return (x, adjusted_y)


def detect_colored_marks(game_bounds, mine_area, color_name, lower_hsv1, upper_hsv1, lower_hsv2=None, upper_hsv2=None, min_area=150):
    """
    Generic function to detect colored marks (red or green) within the mine area only.
    """
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    
    # Crop to mine area only
    mx, my, mw, mh = mine_area
    mine_region = screenshot_cv[my:my+mh, mx:mx+mw]
    
    # Convert to HSV
    hsv = cv2.cvtColor(mine_region, cv2.COLOR_BGR2HSV)
    
    # Color detection
    mask1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
    
    if lower_hsv2 is not None and upper_hsv2 is not None:
        mask2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
        color_mask = cv2.bitwise_or(mask1, mask2)
    else:
        color_mask = mask1
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get all colored region centers
    colored_points = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                colored_points.append((cx, cy))
    
    if len(colored_points) == 0:
        return [], color_mask
    
    # Cluster nearby detections
    colored_array = np.array(colored_points)
    clustering = DBSCAN(eps=45, min_samples=1).fit(colored_array)
    
    # Calculate centroid of each cluster
    cluster_sizes = []
    for label in set(clustering.labels_):
        cluster_points = colored_array[clustering.labels_ == label]
        centroid = cluster_points.mean(axis=0)
        cluster_sizes.append((len(cluster_points), int(centroid[0]) + mx, int(centroid[1]) + my))
    
    # Limit to maximum 4 marks
    if len(cluster_sizes) > 4:
        cluster_sizes.sort(reverse=True)
        colored_centers = [(pos[1], pos[2]) for pos in cluster_sizes[:4]]
    else:
        colored_centers = [(pos[1], pos[2]) for pos in cluster_sizes]
    
    # Sort marks by position
    colored_centers.sort(key=lambda p: (p[1], p[0]))
    
    return colored_centers, color_mask


def detect_red_marks(game_bounds, mine_area):
    """
    Detects red X marks (mines to avoid) in the mine area only.
    """
    lower_red1 = np.array([0, 120, 120])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 120])
    upper_red2 = np.array([180, 255, 255])
    
    return detect_colored_marks(game_bounds, mine_area, "red", lower_red1, upper_red1, lower_red2, upper_red2)


def detect_green_marks(game_bounds, mine_area):
    """
    Detects green checkmarks (successful attacks) in the mine area only.
    """
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])
    
    return detect_colored_marks(game_bounds, mine_area, "green", lower_green, upper_green, min_area=100)


def wait_for_button(button_color, timeout=2.5, search_area=None):
    """
    Wait for a button to appear, checking repeatedly until timeout.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        button = find_button(button_color, search_area)
        
        if button:
            return button
        
        time.sleep(0.1)  # Wait 100ms before retry
    
    return None


def find_button(button_color='red', search_area=None):
    """
    Find a button on screen by color.
    """
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    
    # Crop to search area if provided
    if search_area:
        x, y, w, h = search_area
        screenshot_cv = screenshot_cv[y:y+h, x:x+w]
        offset_x, offset_y = x, y
    else:
        offset_x, offset_y = 0, 0
    
    # Convert to HSV
    hsv = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGR2HSV)
    
    if button_color == 'red':
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        min_area = 1000
    elif button_color == 'green':
        lower_green = np.array([40, 80, 80])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        min_area = 1000
    elif button_color == 'yellow':
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        min_area = 1000
    elif button_color == 'blue':
        lower_blue = np.array([95, 120, 150])
        upper_blue = np.array([115, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        min_area = 3000
    else:
        return None
    
    # Clean up
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    if area > min_area:
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + offset_x
            cy = int(M["m01"] / M["m00"]) + offset_y
            return (cx, cy)
    
    return None


def safe_click(x, y):
    """
    Perform a quick click.
    """
    pyautogui.click(x, y)
    time.sleep(0.15)


def search_for_new_mines():
    """
    Click the blue SEARCH button to find new mines.
    """
    # Get game window bounds to restrict search to bottom area
    game_bounds, _ = find_game_window()
    if game_bounds:
        x, y, w, h = game_bounds
        search_area = (x, y + int(h * 0.7), w, int(h * 0.3))
    else:
        search_area = None
    
    # Look for the blue SEARCH button
    search_button = wait_for_button('blue', timeout=2.5, search_area=search_area)
    
    if search_button is None:
        print("✗ ERROR: SEARCH button not found")
        return False
    
    search_x, search_y = search_button
    safe_click(search_x, search_y)
    
    # Wait for search to complete
    time.sleep(1.0)
    
    # Verify new mines appeared
    results = detect_mines()
    if results and len(results['red_mines']) > 0:
        return True
    else:
        return False

def return_troops_from_mine(mine_position, mine_name):
    """
    Return troops from a green mine.
    """
    # Adjust position - click on the mine structure below the green mark
    adjusted_position = adjust_mine_click_position(mine_position, offset_y=30)
    
    # Click on the green mine
    mine_x, mine_y = adjusted_position
    safe_click(mine_x, mine_y)
    time.sleep(0.2)
    
    # Get dialog area for button search
    game_bounds, _ = find_game_window()
    if game_bounds:
        x, y, w, h = game_bounds
        dialog_x = x + int(w * 0.2)
        dialog_y = y + int(h * 0.25)
        dialog_w = int(w * 0.6)
        dialog_h = int(h * 0.5)
        search_area = (dialog_x, dialog_y, dialog_w, dialog_h)
    else:
        search_area = None
    
    # Wait for and click the yellow RETURN button
    return_button = wait_for_button('yellow', timeout=2.5, search_area=search_area)
    
    if return_button is None:
        return False
    
    return_x, return_y = return_button
    safe_click(return_x, return_y)
    time.sleep(0.2)
    
    stats['troops_returned'] += 1
    return True


def attack_mine(mine_position, mine_name):
    """
    Attack a red mine at the given position.
    """
    stats['total_attacks'] += 1
    
    # Adjust position - click on the mine structure below the red X mark
    adjusted_position = adjust_mine_click_position(mine_position, offset_y=30)
    
    # Click on the mine
    mine_x, mine_y = adjusted_position
    safe_click(mine_x, mine_y)
    time.sleep(0.2)
    
    # Get dialog area for button search
    game_bounds, _ = find_game_window()
    if game_bounds:
        x, y, w, h = game_bounds
        dialog_x = x + int(w * 0.2)
        dialog_y = y + int(h * 0.25)
        dialog_w = int(w * 0.6)
        dialog_h = int(h * 0.5)
        search_area = (dialog_x, dialog_y, dialog_w, dialog_h)
    else:
        search_area = None
    
    # Wait for and click the red ATTACK button
    attack_button = wait_for_button('red', timeout=2.5, search_area=search_area)
    
    if attack_button is None:
        print(f"✗ {mine_name}: FAILED (ATTACK button not found)")
        stats['failed_attacks'] += 1
        return False
    
    attack_x, attack_y = attack_button
    safe_click(attack_x, attack_y)
    time.sleep(0.2)
    
    # Wait for and click the green DEPART button
    depart_button = wait_for_button('green', timeout=2.5, search_area=search_area)
    
    if depart_button is None:
        print(f"✗ {mine_name}: FAILED (DEPART button not found)")
        stats['failed_attacks'] += 1
        return False
    
    depart_x, depart_y = depart_button
    safe_click(depart_x, depart_y)
    time.sleep(1.0)
    
    # Check if mine turned green (successful) or stayed red (failed)
    game_bounds, _ = find_game_window()
    if game_bounds is None:
        print(f"✗ {mine_name}: FAILED (could not verify result)")
        stats['failed_attacks'] += 1
        return False
    
    mine_area = get_mine_area(game_bounds)
    green_centers, _ = detect_green_marks(game_bounds, mine_area)
    
    # Check if the original mine position now has a green mark nearby
    original_x, original_y = mine_position
    for green_pos in green_centers:
        distance = np.sqrt((original_x - green_pos[0])**2 + (original_y - green_pos[1])**2)
        if distance < 60:
            print(f"✓ {mine_name}: SUCCESS")
            stats['successful_attacks'] += 1
            return True
    
    print(f"✗ {mine_name}: FAILED")
    stats['failed_attacks'] += 1
    return False


def classify_mines(centers, prefix):
    """
    Label mines simply as prefix1, prefix2, prefix3, prefix4.
    """
    if len(centers) == 0:
        return []
    
    results = []
    for i, pos in enumerate(centers, 1):
        results.append({
            'position_name': f'{prefix}{i}',
            'position': pos
        })
    
    return results


def detect_mines():
    """
    Detect all red and green mines.
    """
    game_bounds, _ = find_game_window()
    
    if game_bounds is None:
        return None
    
    mine_area = get_mine_area(game_bounds)
    
    # Detect red and green marks
    red_centers, _ = detect_red_marks(game_bounds, mine_area)
    red_results = classify_mines(red_centers, 'red_mine')
    
    green_centers, _ = detect_green_marks(game_bounds, mine_area)
    green_results = classify_mines(green_centers, 'green_mine')
    
    return {
        'red_mines': red_results,
        'green_mines': green_results
    }


def print_statistics(username, power):
    """
    Print current statistics.
    """
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Username:            {username}")
    print(f"Power & Troop type:  {power}")
    print(f"Start Time:          {stats['start_time']}")
    print(f"End Time:            {stats['end_time']}")
    print(f"Total Attacks:       {stats['total_attacks']}")
    print(f"Successful Attacks:  {stats['successful_attacks']} ({stats['successful_attacks']/max(stats['total_attacks'], 1)*100:.1f}%)")
    print(f"Failed Attacks:      {stats['failed_attacks']} ({stats['failed_attacks']/max(stats['total_attacks'], 1)*100:.1f}%)")
    print(f"Searches Performed:  {stats['searches_performed']}")
    print(f"Troops Returned:     {stats['troops_returned']}")
    print("="*70)

def hit_mines(username=None, power=None, max_searches=30):
    failed_mines = set()
    
    while True:
        # Detect current mine status
        results = detect_mines()
        
        if results is None:
            print("✗ ERROR: Could not detect mines!")
            break
        
        red_mines = results['red_mines']
        green_mines = results['green_mines']
        
        print(f"\n[{len(red_mines)} RED, {len(green_mines)} GREEN, Search {stats['searches_performed']}/{max_searches}]")
        
        # Filter out failed mines
        attackable_mines = [m for m in red_mines if m['position_name'] not in failed_mines]
        
        if not attackable_mines:
            # No more attackable mines in current batch
            if len(red_mines) == 0:
                # All mines cleared successfully
                print("\n✓ All current red mines cleared!")
            else:
                # Some mines remain but all have failed
                print(f"\n✗ All remaining {len(red_mines)} red mines have failed.")
            
            # Check if we can search for more
            if max_searches == 0 or stats['searches_performed'] < max_searches:
                print(f"→ Searching for new mines...")
                
                # First, return all troops from green mines
                if green_mines:
                    for green_mine in green_mines:
                        return_troops_from_mine(green_mine['position'], green_mine['position_name'])
                        time.sleep(0.2)
                
                # Now search for new mines
                search_success = search_for_new_mines()
                stats['searches_performed'] += 1  # Increment here
                
                if search_success:
                    # Clear failed mines list for new set
                    failed_mines.clear()
                    print("✓ New mines found!\n")
                    
                    # Continue attacking the new mines even if we've hit max_searches
                    # The next iteration will attack them
                    continue
                else:
                    print("✗ No new mines found. Stopping.\n")
                    break
            else:
                # Reached max searches and no attackable mines
                print(f"\n✗ Reached maximum searches ({max_searches}). Stopping.\n")
                break
        
        # Attack the first available red mine
        target_mine = attackable_mines[0]
        success = attack_mine(target_mine['position'], target_mine['position_name'])
        
        if success:
            # Check if there are still red mines remaining
            time.sleep(0.2)
            results = detect_mines()
            if results and len(results['red_mines']) > 0:
                # Return troops from all green mines
                for green_mine in results['green_mines']:
                    return_troops_from_mine(green_mine['position'], green_mine['position_name'])
                    time.sleep(0.2)
        else:
            # Mark this mine as failed
            failed_mines.add(target_mine['position_name'])
        
        # Small delay before next iteration
        time.sleep(0.2)

def attack_all_red_mines(username=None, power=None, max_searches=30):
    """
    Main function to attack all red mines and manage troops.
    """
    print("\n" + "="*70)
    print("STARTING AUTOMATED MINE ATTACK SEQUENCE")
    print(f"Player: {username}")
    print(f"Power: {power}")
    print(f"Maximum searches allowed: {max_searches}")
    print("="*70 + "\n")
    
    hit_mines(username, power, max_searches)
    
    # Final summary
    stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_statistics(username, power)

def main():
    parser = argparse.ArgumentParser(
        description='Automated Mine Attack System for Whitehill',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python attack.py -u John -p 484Fire
  python attack.py -u Jane -p 500Archer -s 50
  python attack.py --username Bob --power 450Earth --searches 0
        '''
    )
    
    parser.add_argument('-u', '--username', 
                       type=str, 
                       required=True,
                       help='Player username')
    
    parser.add_argument('-p', '--power', 
                       type=str, 
                       required=True,
                       help='Top troop power and type (e.g., 484Fire)')
    
    parser.add_argument('-s', '--searches', 
                       type=int, 
                       default=30,
                       help='Maximum number of searches (default: 30, use 0 for unlimited)')
    
    parser.add_argument('-d', '--delay',
                       type=int,
                       default=2,
                       help='Countdown delay in seconds before starting (default: 2)')
    
    args = parser.parse_args()
    
    # Record start time
    stats['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("="*70)
    print("Mine Detection & Attack System")
    print("="*70)
    print(f"Player:   {args.username}")
    print(f"Power:    {args.power}")
    print(f"Searches: {args.searches if args.searches > 0 else 'Unlimited'}")
    print(f"\nStarting in {args.delay} seconds... Make sure game window is visible!")
    print("="*70)
    
    time.sleep(args.delay)
    
    # Run attack sequence
    attack_all_red_mines(args.username, args.power, max_searches=args.searches)
    
    # Save to CSV
    save_statistics_to_csv(args.username, args.power)


if __name__ == "__main__":
    main()