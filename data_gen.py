import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_data(num_users=100, num_days=30):
    """Generates synthetic log data for insider threat detection."""
    
    np.random.seed(42)
    random.seed(42)
    
    departments = ['Sales', 'Engineering', 'HR', 'Finance', 'IT']
    activity_types = ['logon', 'usb', 'email', 'http']
    
    users = [f'user_{i:03d}' for i in range(num_users)]
    user_depts = {u: random.choice(departments) for u in users}
    
    logs = []
    
    start_date = datetime.now() - timedelta(days=num_days)
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        is_weekend = current_date.weekday() >= 5
        
        for user in users:
            # Base probability of activity
            if is_weekend:
                if random.random() > 0.1: # 10% chance of working on weekend
                    continue
            
            # Number of activities for this user today
            num_activities = np.random.poisson(10)
            
            for _ in range(num_activities):
                # Timestamp
                hour = np.random.normal(14, 4) # Normal distribution centered around 2 PM
                hour = max(0, min(23, hour))
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                timestamp = current_date.replace(hour=int(hour), minute=minute, second=second)
                
                act_type = random.choice(activity_types)
                
                # Volume (MB) - only relevant for some activities, but we'll populate for all
                if act_type in ['usb', 'email', 'http']:
                    volume_mb = np.random.exponential(5) # Exponential distribution for file sizes
                else:
                    volume_mb = 0
                
                logs.append({
                    'user_id': user,
                    'timestamp': timestamp,
                    'dept': user_depts[user],
                    'activity_type': act_type,
                    'volume_mb': volume_mb
                })
    
    # Inject Anomalies (Insider Threats)
    # Pick 5 random users to be "insiders"
    insiders = random.sample(users, 5)
    print(f"Injecting anomalies for: {insiders}")
    
    # Add is_insider column to logs (default False)
    for log in logs:
        log['is_insider'] = log['user_id'] in insiders

    for insider in insiders:
        # Add suspicious activity in the last 5 days
        for day in range(num_days - 5, num_days):
            current_date = start_date + timedelta(days=day)
            
            # 1. After hours activity
            for _ in range(5):
                hour = random.randint(0, 5) # 12 AM to 5 AM
                timestamp = current_date.replace(hour=hour, minute=random.randint(0, 59))
                logs.append({
                    'user_id': insider,
                    'timestamp': timestamp,
                    'dept': user_depts[insider],
                    'activity_type': 'usb', # Suspicious USB usage
                    'volume_mb': np.random.uniform(50, 500), # Large volume
                    'is_insider': True
                })
            
            # 2. High volume data exfiltration
            logs.append({
                'user_id': insider,
                'timestamp': current_date.replace(hour=10),
                'activity_type': 'http',
                'dept': user_depts[insider],
                'volume_mb': np.random.uniform(1000, 5000), # Huge upload
                'is_insider': True
            })

    df = pd.DataFrame(logs)
    return df

if __name__ == "__main__":
    print("Generating synthetic logs...")
    raw_logs = generate_data()
    print(f"Generated {len(raw_logs)} logs.")
    
    print("Saving to CSV...")
    import os
    os.makedirs('data', exist_ok=True)
    raw_logs.to_csv('data/synthetic_logs.csv', index=False)
    print("Done.")
