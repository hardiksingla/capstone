import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import math

class ImprovedBehaviorExtractor:
    """Improved feature extraction with cross-platform support"""
    
    def __init__(self):
        self.features = {}
    
    def load_session(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def extract_all_features(self, session_data):
        """Extract comprehensive features"""
        events = session_data['events']
        duration = session_data.get('duration', 0)
        
        features = {
            'session_id': session_data['sessionId'],
            'duration_seconds': duration / 1000 if duration > 0 else 0,
        }
        
        # Core feature extraction
        features.update(self._extract_navigation_features(events))
        features.update(self._extract_editing_features_enhanced(events))
        features.update(self._extract_typing_rhythm_advanced(events))
        features.update(self._extract_mouse_editing(events))
        features.update(self._extract_code_evolution_pattern(events))
        features.update(self._extract_sequence_patterns(events))
        features.update(self._extract_workflow_patterns(events))
        
        # Label
        features['label'] = 1 if session_data.get('label') == 'suspicious' else 0
        features['suspicion_type'] = session_data.get('suspicionType', 'normal')
        
        return features
    
    def _extract_navigation_features(self, events):
        """Arrow key usage patterns"""
        keydowns = [e for e in events if e['type'] == 'keydown']
        
        arrow_up = sum(1 for e in keydowns if e.get('key') == 'ArrowUp')
        arrow_down = sum(1 for e in keydowns if e.get('key') == 'ArrowDown')
        arrow_left = sum(1 for e in keydowns if e.get('key') == 'ArrowLeft')
        arrow_right = sum(1 for e in keydowns if e.get('key') == 'ArrowRight')
        
        total_arrows = arrow_up + arrow_down + arrow_left + arrow_right
        vertical_nav = arrow_up + arrow_down
        horizontal_nav = arrow_left + arrow_right
        
        # Navigation density
        duration_min = (events[-1]['timestamp'] - events[0]['timestamp']) / 60000 if len(events) > 1 else 1
        nav_density = total_arrows / duration_min if duration_min > 0 else 0
        
        # Navigation patterns
        total_keys = len(keydowns)
        nav_ratio = total_arrows / total_keys if total_keys > 0 else 0
        
        # Vertical vs horizontal ratio
        vh_ratio = vertical_nav / horizontal_nav if horizontal_nav > 0 else (vertical_nav if vertical_nav > 0 else 0)
        
        return {
            'arrow_total': total_arrows,
            'arrow_vertical': vertical_nav,
            'arrow_horizontal': horizontal_nav,
            'arrow_density_per_min': nav_density,
            'arrow_to_keystroke_ratio': nav_ratio,
            'vertical_horizontal_ratio': vh_ratio
        }
    
    def _extract_editing_features_enhanced(self, events):
        """ENHANCED: Better deletion detection with ctrl/meta modifiers"""
        keydowns = [e for e in events if e['type'] == 'keydown']
        char_keys = sum(1 for e in keydowns if e.get('key') == 'char')
        
        # Enhanced backspace counting
        weighted_backspaces = 0
        raw_backspaces = 0
        delete_keys = 0
        
        for e in keydowns:
            if e.get('key') == 'Backspace':
                raw_backspaces += 1
                # Ctrl+Backspace (Windows) or Meta+Backspace (Mac) = delete word = 5 backspaces
                if e.get('ctrlKey') or e.get('metaKey'):
                    weighted_backspaces += 5
                else:
                    weighted_backspaces += 1
            
            elif e.get('key') == 'Delete':
                delete_keys += 1
                # Ctrl+Delete or Meta+Delete = delete word
                if e.get('ctrlKey') or e.get('metaKey'):
                    weighted_backspaces += 5
                else:
                    weighted_backspaces += 1
        
        total_deletions = weighted_backspaces
        
        # Ratios with weighted deletions
        deletion_ratio = total_deletions / len(keydowns) if len(keydowns) > 0 else 0
        deletion_to_char_ratio = total_deletions / char_keys if char_keys > 0 else 0
        
        # Consecutive deletion sequences
        consecutive_deletion_sequences = 0
        max_consecutive_deletions = 0
        current_sequence = 0
        
        for i, e in enumerate(keydowns):
            is_deletion = e.get('key') in ['Backspace', 'Delete']
            if is_deletion:
                if i > 0 and keydowns[i-1].get('key') in ['Backspace', 'Delete']:
                    current_sequence += 1
                else:
                    if current_sequence > 0:
                        consecutive_deletion_sequences += 1
                        max_consecutive_deletions = max(max_consecutive_deletions, current_sequence)
                    current_sequence = 1
            
            if current_sequence > 0 and i == len(keydowns) - 1:
                consecutive_deletion_sequences += 1
                max_consecutive_deletions = max(max_consecutive_deletions, current_sequence)
        
        # MERGED: Modifier key usage (Ctrl on Windows/Linux, Meta/Cmd on Mac)
        modifier_usage = sum(1 for e in keydowns if e.get('ctrlKey') or e.get('metaKey'))
        
        return {
            'deletion_count_weighted': total_deletions,
            'deletion_ratio': deletion_ratio,
            'deletion_to_char_ratio': deletion_to_char_ratio,
            'raw_backspace_count': raw_backspaces,
            'delete_key_count': delete_keys,
            'deletion_sequences': consecutive_deletion_sequences,
            'max_consecutive_deletions': max_consecutive_deletions,
            'char_key_count': char_keys,
            'modifier_key_usage': modifier_usage
        }
    
    def _extract_typing_rhythm_advanced(self, events):
        """Advanced typing rhythm with burst detection"""
        keydowns = [e for e in events if e['type'] == 'keydown' and e.get('key') == 'char']
        
        if len(keydowns) < 2:
            return {
                'typing_variance': 0,
                'typing_std': 0,
                'mean_keystroke_interval': 0,
                'median_keystroke_interval': 0,
                'pause_short_count': 0,
                'pause_medium_count': 0,
                'pause_long_count': 0,
                'typing_burst_count': 0,
                'typing_consistency_score': 0,
                'rhythm_irregularity': 0
            }
        
        # Inter-keystroke intervals
        intervals = []
        for i in range(1, len(keydowns)):
            interval = keydowns[i]['timestamp'] - keydowns[i-1]['timestamp']
            if interval < 10000:  # Ignore pauses > 10s
                intervals.append(interval)
        
        if len(intervals) == 0:
            return {
                'typing_variance': 0,
                'typing_std': 0,
                'mean_keystroke_interval': 0,
                'median_keystroke_interval': 0,
                'pause_short_count': 0,
                'pause_medium_count': 0,
                'pause_long_count': 0,
                'typing_burst_count': 0,
                'typing_consistency_score': 0,
                'rhythm_irregularity': 0
            }
        
        # Statistical measures
        mean_interval = np.mean(intervals)
        median_interval = np.median(intervals)
        std_interval = np.std(intervals)
        
        # Pause categorization
        pause_short = sum(1 for i in intervals if 1000 < i <= 3000)
        pause_medium = sum(1 for i in intervals if 3000 < i <= 5000)
        pause_long = sum(1 for i in intervals if i > 5000)
        
        # Typing bursts
        burst_count = 0
        in_burst = False
        
        for interval in intervals:
            if interval < 250:  # Fast typing
                if not in_burst:
                    burst_count += 1
                    in_burst = True
            else:
                in_burst = False
        
        # Consistency score (coefficient of variation)
        consistency_score = std_interval / mean_interval if mean_interval > 0 else 0
        
        # Rhythm irregularity (higher = more human-like)
        mad = np.median([abs(i - median_interval) for i in intervals])
        rhythm_irregularity = mad / median_interval if median_interval > 0 else 0
        
        return {
            'typing_variance': std_interval ** 2,
            'typing_std': std_interval,
            'mean_keystroke_interval': mean_interval,
            'median_keystroke_interval': median_interval,
            'pause_short_count': pause_short,
            'pause_medium_count': pause_medium,
            'pause_long_count': pause_long,
            'typing_burst_count': burst_count,
            'typing_consistency_score': consistency_score,
            'rhythm_irregularity': rhythm_irregularity
        }
    
    def _extract_mouse_editing(self, events):
        """Mouse interaction patterns"""
        clicks = [e for e in events if e['type'] == 'click']
        mouse_moves = [e for e in events if e['type'] == 'mousemove']
        
        # Editor area clicks (heuristic)
        editor_clicks = sum(1 for e in clicks if 300 < e.get('x', 0) < 1000 and 200 < e.get('y', 0) < 800)
        
        # Click patterns
        duration_min = (events[-1]['timestamp'] - events[0]['timestamp']) / 60000 if len(events) > 1 else 1
        click_density = editor_clicks / duration_min if duration_min > 0 else 0
        
        # Mouse movement activity
        mouse_activity = len(mouse_moves) / duration_min if duration_min > 0 else 0
        
        return {
            'editor_click_count': editor_clicks,
            'click_density_per_min': click_density,
            'total_click_count': len(clicks),
            'mouse_movement_density': mouse_activity
        }
    
    def _extract_code_evolution_pattern(self, events):
        """Code growth patterns"""
        snapshots = [e for e in events if e['type'] == 'codeSnapshot']
        
        if len(snapshots) < 2:
            return {
                'snapshot_count': len(snapshots),
                'code_growth_variance': 0,
                'sudden_jumps': 0,
                'deletion_events': 0,
                'growth_linearity_score': 0,
                'code_volatility': 0
            }
        
        # Track code length changes
        lengths = [s.get('length', 0) for s in snapshots]
        changes = [lengths[i] - lengths[i-1] for i in range(1, len(lengths))]
        
        # Variance in code growth
        growth_variance = np.std(changes) if len(changes) > 1 else 0
        
        # Sudden large additions
        sudden_jumps = sum(1 for c in changes if c > 50)
        
        # Deletions (negative changes)
        deletion_events = sum(1 for c in changes if c < -10)
        
        # Linearity score
        positive_changes = sum(1 for c in changes if c > 0)
        linearity_score = positive_changes / len(changes) if len(changes) > 0 else 0
        
        # Code volatility (how much code changes)
        total_change = sum(abs(c) for c in changes)
        volatility = total_change / len(changes) if len(changes) > 0 else 0
        
        return {
            'snapshot_count': len(snapshots),
            'code_growth_variance': growth_variance,
            'sudden_jumps': sudden_jumps,
            'deletion_events': deletion_events,
            'growth_linearity_score': linearity_score,
            'code_volatility': volatility
        }
    
    def _extract_sequence_patterns(self, events):
        """Sequential behavior patterns"""
        keydowns = [e for e in events if e['type'] == 'keydown']
        
        if len(keydowns) < 3:
            return {
                'edit_navigation_switches': 0,
                'rapid_context_switches': 0
            }
        
        # Count switches between typing and navigation
        switches = 0
        last_was_nav = None
        
        for e in keydowns:
            is_nav = 'Arrow' in e.get('key', '') or e.get('key') in ['Home', 'End', 'PageUp', 'PageDown']
            if last_was_nav is not None and last_was_nav != is_nav:
                switches += 1
            last_was_nav = is_nav
        
        # Rapid context switches (nav -> type -> nav within 2 seconds)
        rapid_switches = 0
        for i in range(2, len(keydowns)):
            if keydowns[i]['timestamp'] - keydowns[i-2]['timestamp'] < 2000:
                k1_nav = 'Arrow' in keydowns[i-2].get('key', '')
                k2_nav = 'Arrow' in keydowns[i-1].get('key', '')
                k3_nav = 'Arrow' in keydowns[i].get('key', '')
                
                if k1_nav and not k2_nav and k3_nav:
                    rapid_switches += 1
        
        return {
            'edit_navigation_switches': switches,
            'rapid_context_switches': rapid_switches
        }
    
    def _extract_workflow_patterns(self, events):
        """High-level workflow indicators using actual copy/paste events"""
        keydowns = [e for e in events if e['type'] == 'keydown']
        
        # Get actual copy and paste events
        copy_events = [e for e in events if e['type'] == 'copy']
        paste_events = [e for e in events if e['type'] == 'paste']
        blur_events = [e for e in events if e['type'] == 'blur']
        focus_events = [e for e in events if e['type'] == 'focus']
        
        copy_ops = len(copy_events)
        paste_ops = len(paste_events)
        
        # Selection operations (Shift + Arrow)
        selection_ops = sum(1 for e in keydowns if e.get('shiftKey') and 'Arrow' in e.get('key', ''))
        
        # Helper function to check for modifier key (Ctrl or Meta)
        def has_modifier(e):
            return e.get('ctrlKey') or e.get('metaKey')
        
        # Undo/Redo
        undo_ops = sum(1 for e in keydowns if has_modifier(e) and e.get('key') in ['z', 'Z'])
        redo_ops = sum(1 for e in keydowns if has_modifier(e) and e.get('key') in ['y', 'Y'])
        
        # Save operations
        save_ops = sum(1 for e in keydowns if has_modifier(e) and e.get('key') in ['s', 'S'])

        run_code_ops = sum(1 for e in keydowns if has_modifier(e) and e.get('key') == "'")

        submit_ops = sum(1 for e in keydowns if has_modifier(e) and e.get('key') == 'Enter')

        # Tab switches (blur count)
        tab_switches = len(blur_events)
        
        # Paste analysis
        paste_lengths = [e.get('length', 0) for e in paste_events]
        paste_lines = [e.get('lines', 0) for e in paste_events]
        
        avg_paste_length = np.mean(paste_lengths) if paste_lengths else 0
        max_paste_length = max(paste_lengths) if paste_lengths else 0
        avg_paste_lines = np.mean(paste_lines) if paste_lines else 0
        max_paste_lines = max(paste_lines) if paste_lines else 0
        
        # Large paste detection (>500 chars or >10 lines)
        large_pastes = sum(1 for e in paste_events if e.get('length', 0) > 500 or e.get('lines', 0) > 10)
        
        # Paste without prior copy (external source - very suspicious)
        pastes_without_copy = 0
        for paste_event in paste_events:
            has_prior_copy = any(copy_event['timestamp'] < paste_event['timestamp'] 
                               for copy_event in copy_events)
            if not has_prior_copy:
                pastes_without_copy += 1
        
        # Time spent outside editor
        time_outside_editor = 0
        for i in range(len(blur_events)):
            blur_time = blur_events[i]['timestamp']
            next_focus = None
            for focus in focus_events:
                if focus['timestamp'] > blur_time:
                    next_focus = focus
                    break
            
            if next_focus:
                time_outside_editor += (next_focus['timestamp'] - blur_time)
        
        time_outside_editor_seconds = time_outside_editor / 1000
        
        # Calculate densities
        duration_min = (events[-1]['timestamp'] - events[0]['timestamp']) / 60000 if len(events) > 1 else 1
        paste_density = paste_ops / duration_min if duration_min > 0 else 0
        tab_switch_density = tab_switches / duration_min if duration_min > 0 else 0
        
        return {
            'selection_operations': selection_ops,
            'copy_operations': copy_ops,
            'paste_operations': paste_ops,
            'undo_operations': undo_ops,
            'redo_operations': redo_ops,
            'save_operations': save_ops,
            'total_editing_operations': selection_ops + undo_ops + redo_ops,
            'run_code_count': run_code_ops,
            'submit_code_count': submit_ops,
            'run_to_submit_ratio': run_code_ops / submit_ops if submit_ops > 0 else run_code_ops,
            'tab_switches': tab_switches,
            'paste_density_per_min': paste_density,
            'tab_switch_density_per_min': tab_switch_density,
            'avg_paste_length': avg_paste_length,
            'max_paste_length': max_paste_length,
            'avg_paste_lines': avg_paste_lines,
            'max_paste_lines': max_paste_lines,
            'large_pastes': large_pastes,
            'pastes_without_copy': pastes_without_copy,
            'time_outside_editor_seconds': time_outside_editor_seconds,
            'paste_to_copy_ratio': paste_ops / copy_ops if copy_ops > 0 else paste_ops
        }
    
    def process_all_sessions(self, session_dirs):
        """Process all sessions"""
        all_features = []
        
        for session_dir in session_dirs:
            if not os.path.exists(session_dir):
                print(f"⚠️  Directory not found: {session_dir}")
                continue
            
            session_files = list(Path(session_dir).glob('*.json'))
            print(f"\n📂 Processing {len(session_files)} sessions from {session_dir}")
            
            for i, filepath in enumerate(session_files):
                try:
                    session_data = self.load_session(filepath)
                    features = self.extract_all_features(session_data)
                    all_features.append(features)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(session_files)} sessions...")
                        
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")
                    continue
        
        df = pd.DataFrame(all_features)
        
        print(f"\n✅ Extracted features from {len(df)} sessions")
        print(f"   Normal sessions: {sum(df['label'] == 0)}")
        print(f"   Suspicious sessions: {sum(df['label'] == 1)}")
        
        return df

def main():
    print("🔧 Starting IMPROVED Feature Extraction (Cross-Platform)...\n")
    
    extractor = ImprovedBehaviorExtractor()
    
    session_dirs = [
        'sessions2/normal',
        'sessions2/suspicious'
    ]
    
    df = extractor.process_all_sessions(session_dirs)
    
    os.makedirs('data/features', exist_ok=True)
    output_path = 'data/features/improved_features.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Features saved to: {output_path}")
    print(f"\n📊 Feature Summary:")
    print(f"   Total features: {len(df.columns) - 3}")
    print(f"   Total sessions: {len(df)}")
    
    # Compare key features
    print(f"\n📈 Key Feature Comparison (Normal vs Suspicious):")
    key_features = [
        'arrow_total',
        'deletion_count_weighted',
        'deletion_ratio',
        'modifier_key_usage',
        'typing_consistency_score',
        'rhythm_irregularity',
        'edit_navigation_switches',
        'paste_operations',
        'tab_switches',
        'avg_paste_length',
        'max_paste_length',
        'large_pastes',
        'pastes_without_copy'
    ]
    
    for feature in key_features:
        if feature in df.columns:
            normal_mean = df[df['label']==0][feature].mean()
            sus_mean = df[df['label']==1][feature].mean()
            diff = abs(normal_mean - sus_mean)
            print(f"\n  {feature}:")
            print(f"    Normal:     {normal_mean:8.3f}")
            print(f"    Suspicious: {sus_mean:8.3f}")
            print(f"    Difference: {diff:8.3f}")

if __name__ == '__main__':
    main()