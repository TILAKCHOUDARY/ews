"""
Driving Performance Score Analyzer
Based on research papers on driver behavior analysis using IMU and GPS data

Analyzes:
- Speed management (acceleration, deceleration, consistency)
- Turning behavior (smooth vs aggressive)
- Event handling (bumps, stops)
- Overall driving smoothness and safety
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


class DrivingScoreAnalyzer:
    """
    Comprehensive driving performance analyzer
    
    Based on research metrics:
    1. Speed Variation Index (SVI)
    2. Acceleration Smoothness Score (ASS)
    3. Turning Quality Index (TQI)
    4. Event Response Score (ERS)
    5. Overall Driving Score (ODS)
    """
    
    def __init__(self, ride_folder):
        """
        Initialize analyzer for a specific ride
        
        Args:
            ride_folder: Path to rideXX folder (e.g., data/ride01)
        """
        self.ride_folder = ride_folder
        self.ride_name = os.path.basename(ride_folder)
        
        # Load data
        self.imu_data = None
        self.events = None
        self.load_data()
        
        # Results
        self.metrics = {}
        self.final_score = 0
        
    def load_data(self):
        """Load IMU data and events from ride folder"""
        try:
            # Load combined_data.csv
            imu_path = os.path.join(self.ride_folder, "combined_data.csv")
            self.imu_data = pd.read_csv(imu_path)
            print(f"✅ Loaded IMU data: {len(self.imu_data)} samples")
            
            # Convert speed from knots to km/h if available
            if 'gps_speed_kn' in self.imu_data.columns:
                self.imu_data['speed_kmh'] = pd.to_numeric(
                    self.imu_data['gps_speed_kn'], errors='coerce'
                ) * 1.852
            else:
                self.imu_data['speed_kmh'] = 0
            
            # Load events.csv if exists
            events_path = os.path.join(self.ride_folder, "events.csv")
            if os.path.exists(events_path):
                self.events = pd.read_csv(events_path)
                print(f"✅ Loaded events: {len(self.events)} events")
            else:
                print("⚠️  No events.csv found")
                self.events = pd.DataFrame(columns=['timestamp', 'event'])
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def analyze_all(self):
        """Run complete driving analysis"""
        print("\n" + "="*60)
        print(f"   🚗 ANALYZING DRIVING PERFORMANCE: {self.ride_name}")
        print("="*60 + "\n")
        
        # Run all analyses
        self.analyze_speed_management()
        self.analyze_acceleration_smoothness()
        self.analyze_turning_quality()
        self.analyze_event_responses()
        
        # Calculate final score
        self.calculate_final_score()
        
        # Generate report
        self.generate_report()
        
        return self.final_score, self.metrics
    
    def analyze_speed_management(self):
        """
        Analyze speed management quality
        
        Metrics:
        - Average speed
        - Speed variation (standard deviation)
        - Excessive speeding instances
        - Speed consistency
        """
        print("📊 Analyzing Speed Management...")
        
        speeds = self.imu_data['speed_kmh'].replace(0, np.nan).dropna()
        
        if len(speeds) == 0:
            print("   ⚠️  No speed data available")
            self.metrics['speed_management'] = {
                'score': 50,
                'avg_speed': 0,
                'speed_variation': 0,
                'consistency': 50
            }
            return
        
        avg_speed = speeds.mean()
        speed_std = speeds.std()
        speed_variation_index = (speed_std / avg_speed * 100) if avg_speed > 0 else 100
        
        # Score: Lower variation = better (0-30% variation = 100 points)
        if speed_variation_index < 30:
            variation_score = 100
        elif speed_variation_index < 50:
            variation_score = 100 - (speed_variation_index - 30) * 2
        else:
            variation_score = max(20, 100 - speed_variation_index)
        
        # Speed consistency (changes per minute)
        speed_changes = np.abs(np.diff(speeds))
        significant_changes = np.sum(speed_changes > 5)  # >5 km/h changes
        consistency_score = max(0, 100 - significant_changes * 2)
        
        overall_speed_score = (variation_score * 0.6 + consistency_score * 0.4)
        
        self.metrics['speed_management'] = {
            'score': round(overall_speed_score, 2),
            'avg_speed': round(avg_speed, 2),
            'speed_variation': round(speed_variation_index, 2),
            'consistency': round(consistency_score, 2)
        }
        
        print(f"   ✓ Average Speed: {avg_speed:.2f} km/h")
        print(f"   ✓ Speed Variation: {speed_variation_index:.2f}%")
        print(f"   ✓ Score: {overall_speed_score:.2f}/100\n")
    
    def analyze_acceleration_smoothness(self):
        """
        Analyze acceleration/deceleration smoothness
        
        Smooth acceleration/braking = safe driving
        Harsh movements = aggressive/unsafe
        """
        print("📊 Analyzing Acceleration Smoothness...")
        
        # Calculate acceleration from IMU data
        ax = self.imu_data['ax_g'].values
        ay = self.imu_data['ay_g'].values
        
        # Total acceleration magnitude
        accel_magnitude = np.sqrt(ax**2 + ay**2)
        
        # Jerk (rate of change of acceleration) - key smoothness metric
        jerk = np.abs(np.diff(accel_magnitude))
        
        # Statistical measures
        avg_jerk = np.mean(jerk)
        max_jerk = np.max(jerk)
        jerk_std = np.std(jerk)
        
        # Harsh events (high jerk)
        harsh_accel = np.sum(jerk > 0.5)  # Harsh acceleration threshold
        moderate_accel = np.sum((jerk > 0.2) & (jerk <= 0.5))
        
        # Score calculation
        # Lower jerk = smoother = better score
        if avg_jerk < 0.1:
            smoothness_score = 100
        elif avg_jerk < 0.2:
            smoothness_score = 90 - (avg_jerk - 0.1) * 100
        elif avg_jerk < 0.3:
            smoothness_score = 70 - (avg_jerk - 0.2) * 200
        else:
            smoothness_score = max(20, 50 - avg_jerk * 100)
        
        # Penalty for harsh events
        harsh_penalty = min(30, harsh_accel * 2)
        final_accel_score = max(0, smoothness_score - harsh_penalty)
        
        self.metrics['acceleration_smoothness'] = {
            'score': round(final_accel_score, 2),
            'avg_jerk': round(avg_jerk, 4),
            'harsh_events': int(harsh_accel),
            'moderate_events': int(moderate_accel)
        }
        
        print(f"   ✓ Average Jerk: {avg_jerk:.4f} g/s")
        print(f"   ✓ Harsh Events: {harsh_accel}")
        print(f"   ✓ Score: {final_accel_score:.2f}/100\n")
    
    def analyze_turning_quality(self):
        """
        Analyze turning behavior quality
        
        Good turning:
        - Reduced speed before turn
        - Smooth rotation
        - Controlled exit
        """
        print("📊 Analyzing Turning Quality...")
        
        if len(self.events) == 0:
            print("   ⚠️  No turn events detected")
            self.metrics['turning_quality'] = {
                'score': 70,
                'turns_analyzed': 0
            }
            return
        
        # Get turn events
        turns = self.events[self.events['event'].isin(['left', 'right'])]
        
        if len(turns) == 0:
            print("   ⚠️  No turn events detected")
            self.metrics['turning_quality'] = {
                'score': 70,
                'turns_analyzed': 0
            }
            return
        
        turn_scores = []
        
        for idx, turn in turns.iterrows():
            turn_score = self._analyze_single_turn(turn)
            if turn_score is not None:
                turn_scores.append(turn_score)
        
        if len(turn_scores) > 0:
            avg_turn_score = np.mean(turn_scores)
            turn_quality = avg_turn_score
        else:
            turn_quality = 70
        
        self.metrics['turning_quality'] = {
            'score': round(turn_quality, 2),
            'turns_analyzed': len(turn_scores),
            'left_turns': len(turns[turns['event'] == 'left']),
            'right_turns': len(turns[turns['event'] == 'right'])
        }
        
        print(f"   ✓ Turns Analyzed: {len(turn_scores)}")
        print(f"   ✓ Score: {turn_quality:.2f}/100\n")
    
    def _analyze_single_turn(self, turn_event):
        """Analyze a single turn event"""
        try:
            # Find turn in IMU data by timestamp
            turn_timestamp = turn_event['timestamp']
            
            # Find index in IMU data
            turn_idx = self.imu_data[self.imu_data['timestamp'] == turn_timestamp].index
            
            if len(turn_idx) == 0:
                return None
            
            turn_idx = turn_idx[0]
            
            # Get data before, during, after turn (±5 samples)
            before_idx = max(0, turn_idx - 5)
            after_idx = min(len(self.imu_data), turn_idx + 5)
            
            turn_data = self.imu_data.iloc[before_idx:after_idx]
            
            # Speed analysis
            speeds = turn_data['speed_kmh'].replace(0, np.nan).dropna()
            if len(speeds) < 3:
                return None
            
            speed_before = speeds.iloc[0] if len(speeds) > 0 else 0
            speed_during = speeds.iloc[len(speeds)//2] if len(speeds) > 2 else 0
            speed_after = speeds.iloc[-1] if len(speeds) > 0 else 0
            
            # Check if speed reduced before turn (good behavior)
            speed_reduction = speed_before - speed_during
            if speed_reduction > 5:  # Reduced by >5 km/h
                speed_score = 100
            elif speed_reduction > 0:
                speed_score = 70 + speed_reduction * 6
            else:
                speed_score = max(30, 70 + speed_reduction * 10)  # Penalty for increasing
            
            # Gyroscope smoothness during turn
            gz = turn_data['gz_dps'].values
            gz_smoothness = 100 - min(50, np.std(gz) * 2)
            
            # Overall turn score
            turn_score = speed_score * 0.6 + gz_smoothness * 0.4
            
            return turn_score
            
        except Exception as e:
            return None
    
    def analyze_event_responses(self):
        """
        Analyze responses to events (bumps, stops)
        
        Good driver:
        - Slows down before bumps
        - Smooth stops
        - Quick recovery
        """
        print("📊 Analyzing Event Responses...")
        
        if len(self.events) == 0:
            print("   ⚠️  No events to analyze")
            self.metrics['event_responses'] = {
                'score': 70,
                'events_analyzed': 0
            }
            return
        
        # Analyze bumps
        bumps = self.events[self.events['event'] == 'bump']
        bump_scores = []
        
        for idx, bump in bumps.iterrows():
            score = self._analyze_bump_response(bump)
            if score is not None:
                bump_scores.append(score)
        
        # Analyze potholes
        potholes = self.events[self.events['event'] == 'pothole']
        pothole_scores = []
        
        for idx, pothole in potholes.iterrows():
            score = self._analyze_pothole_response(pothole)
            if score is not None:
                pothole_scores.append(score)
        
        # Analyze stops
        stops = self.events[self.events['event'] == 'stop']
        stop_scores = []
        
        for idx, stop in stops.iterrows():
            score = self._analyze_stop_response(stop)
            if score is not None:
                stop_scores.append(score)
        
        # Overall event response score
        all_scores = bump_scores + pothole_scores + stop_scores
        if len(all_scores) > 0:
            event_score = np.mean(all_scores)
        else:
            event_score = 70
        
        self.metrics['event_responses'] = {
            'score': round(event_score, 2),
            'events_analyzed': len(all_scores),
            'bumps': len(bump_scores),
            'potholes': len(pothole_scores),
            'stops': len(stop_scores)
        }
        
        print(f"   ✓ Events Analyzed: {len(all_scores)}")
        print(f"   ✓ Score: {event_score:.2f}/100\n")
    
    def _analyze_bump_response(self, bump_event):
        """Analyze response to bump"""
        try:
            bump_timestamp = bump_event['timestamp']
            bump_idx = self.imu_data[self.imu_data['timestamp'] == bump_timestamp].index
            
            if len(bump_idx) == 0:
                return None
            
            bump_idx = bump_idx[0]
            
            # Get speed before bump
            before_idx = max(0, bump_idx - 3)
            speed_before = self.imu_data.iloc[before_idx:bump_idx]['speed_kmh'].replace(0, np.nan).mean()
            
            if pd.isna(speed_before) or speed_before == 0:
                return 70  # No speed data, neutral score
            
            # Lower speed before bump = better
            if speed_before < 20:
                return 100
            elif speed_before < 40:
                return 90 - (speed_before - 20)
            else:
                return max(30, 70 - (speed_before - 40))
            
        except Exception:
            return None
    
    def _analyze_pothole_response(self, pothole_event):
        """Analyze response to pothole - similar to bump but more critical"""
        try:
            pothole_timestamp = pothole_event['timestamp']
            pothole_idx = self.imu_data[self.imu_data['timestamp'] == pothole_timestamp].index
            
            if len(pothole_idx) == 0:
                return None
            
            pothole_idx = pothole_idx[0]
            
            # Get speed before pothole
            before_idx = max(0, pothole_idx - 3)
            speed_before = self.imu_data.iloc[before_idx:pothole_idx]['speed_kmh'].replace(0, np.nan).mean()
            
            if pd.isna(speed_before) or speed_before == 0:
                return 70  # No speed data, neutral score
            
            # Potholes are more dangerous - stricter scoring
            if speed_before < 15:
                return 100
            elif speed_before < 30:
                return 95 - (speed_before - 15) * 2
            else:
                return max(20, 65 - (speed_before - 30))
            
        except Exception:
            return None
    
    def _analyze_stop_response(self, stop_event):
        """Analyze stopping behavior"""
        try:
            stop_timestamp = stop_event['timestamp']
            stop_idx = self.imu_data[self.imu_data['timestamp'] == stop_timestamp].index
            
            if len(stop_idx) == 0:
                return None
            
            stop_idx = stop_idx[0]
            
            # Get deceleration profile
            before_idx = max(0, stop_idx - 5)
            stop_data = self.imu_data.iloc[before_idx:stop_idx]
            
            # Check deceleration smoothness
            ax = stop_data['ax_g'].values
            decel_smoothness = 100 - min(50, np.std(ax) * 50)
            
            return decel_smoothness
            
        except Exception:
            return None
    
    def calculate_final_score(self):
        """
        Calculate final driving score using weighted formula
        
        Weights based on importance:
        - Speed Management: 25%
        - Acceleration Smoothness: 30%
        - Turning Quality: 25%
        - Event Responses: 20%
        """
        print("📊 Calculating Final Score...\n")
        
        weights = {
            'speed_management': 0.25,
            'acceleration_smoothness': 0.30,
            'turning_quality': 0.25,
            'event_responses': 0.20
        }
        
        total_score = 0
        for metric, weight in weights.items():
            score = self.metrics.get(metric, {}).get('score', 70)
            total_score += score * weight
        
        self.final_score = round(total_score, 2)
        
        # Determine rating
        if self.final_score >= 90:
            rating = "EXCELLENT"
            emoji = "🌟"
        elif self.final_score >= 80:
            rating = "VERY GOOD"
            emoji = "⭐"
        elif self.final_score >= 70:
            rating = "GOOD"
            emoji = "✅"
        elif self.final_score >= 60:
            rating = "AVERAGE"
            emoji = "👍"
        else:
            rating = "NEEDS IMPROVEMENT"
            emoji = "⚠️"
        
        self.metrics['final'] = {
            'score': self.final_score,
            'rating': rating,
            'emoji': emoji
        }
    
    def generate_report(self):
        """Generate and save detailed report"""
        print("="*60)
        print(f"   {self.metrics['final']['emoji']} DRIVING PERFORMANCE REPORT")
        print("="*60)
        print(f"\n🚗 Ride: {self.ride_name}")
        print(f"📊 Samples: {len(self.imu_data)}")
        print(f"🎯 Events: {len(self.events)}\n")
        
        print("📈 DETAILED SCORES:")
        print("-"*60)
        
        for key, data in self.metrics.items():
            if key != 'final':
                title = key.replace('_', ' ').title()
                score = data.get('score', 0)
                print(f"  {title:30s}: {score:6.2f}/100")
        
        print("-"*60)
        print(f"  {'FINAL SCORE':30s}: {self.final_score:6.2f}/100")
        print(f"  {'RATING':30s}: {self.metrics['final']['rating']}")
        print("="*60 + "\n")
        
        # Save to JSON
        report_path = os.path.join(self.ride_folder, "driving_score_report.json")
        report_data = {
            'ride': self.ride_name,
            'timestamp': datetime.now().isoformat(),
            'samples': len(self.imu_data),
            'events': len(self.events),
            'metrics': self.metrics,
            'final_score': self.final_score
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"💾 Report saved: {report_path}\n")
        
        return report_path


def main():
    """Main function to run analyzer"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python driving_score_analyzer.py <ride_folder>")
        print("Example: python driving_score_analyzer.py ../data/ride01")
        sys.exit(1)
    
    ride_folder = sys.argv[1]
    
    if not os.path.exists(ride_folder):
        print(f"❌ Error: Folder not found: {ride_folder}")
        sys.exit(1)
    
    # Run analysis
    analyzer = DrivingScoreAnalyzer(ride_folder)
    final_score, metrics = analyzer.analyze_all()
    
    print(f"\n✅ Analysis complete! Final Score: {final_score}/100\n")


if __name__ == "__main__":
    main()
