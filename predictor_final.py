import pickle
import json
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from feature_extractor_v4 import ImprovedBehaviorExtractor

class LinearTypingDetector:
    """Detect linear typing patterns using trained model"""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        self.extractor = ImprovedBehaviorExtractor()
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model and metadata"""
        print("📦 Loading trained model...")
        
        # Load model
        with open(f'{self.model_dir}/best_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(f'{self.model_dir}/feature_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature names
        with open(f'{self.model_dir}/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        # Load metadata
        try:
            with open(f'{self.model_dir}/model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = None
        
        model_type = self.metadata['model_type'] if self.metadata else type(self.model).__name__
        print(f"   ✓ Model loaded: {model_type}")
        print(f"   ✓ Features: {len(self.feature_names)}")
        
        if self.metadata and 'best_model_metrics' in self.metadata:
            f1_mean = self.metadata['best_model_metrics']['f1']['mean']
            print(f"   ✓ CV F1-Score: {f1_mean:.3f}")
    
    def predict_session(self, session_path):
        """Predict on a single session"""
        # Extract features
        session_data = self.extractor.load_session(session_path)
        features = self.extractor.extract_all_features(session_data)
        
        # Prepare feature vector
        X = np.array([[features[name] for name in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        suspicion_score = probability[1]
        
        # Analyze indicators
        indicators = self._analyze_indicators(features)
        
        return {
            'session_id': features['session_id'],
            'prediction': 'LINEAR_TYPING' if prediction == 1 else 'GENUINE_SOLVING',
            'suspicion_score': float(suspicion_score),
            'confidence': float(max(probability)),
            'risk_level': self._get_risk_level(suspicion_score),
            'key_indicators': indicators,
            'behavioral_stats': self._extract_stats(features)
        }
    
    def _extract_stats(self, features):
        """Extract key statistics"""
        return {
            'arrow_keys': features['arrow_total'],
            'backspaces': features.get('raw_backspace_count', 0),
            'deletion_ratio': features['deletion_ratio'],
            'typing_variance': features['typing_variance'],
            'editor_clicks': features['editor_click_count'],
            'run_count': features.get('run_code_count', 0),
            'submit_count': features.get('submit_code_count', 0),
            'edit_switches': features.get('edit_navigation_switches', 0),
            'modifier_usage': features.get('modifier_key_usage', 0)
        }
    
    def _analyze_indicators(self, features):
        """Analyze behavioral indicators"""
        indicators = []
        
        # Arrow keys
        arrow_total = features['arrow_total']
        if arrow_total < 5:
            indicators.append({
                'type': 'critical',
                'feature': 'Arrow Keys',
                'value': arrow_total,
                'message': f'Very few arrow keys ({arrow_total})',
                'detail': 'No code navigation detected - linear typing pattern'
            })
        elif arrow_total < 10:
            indicators.append({
                'type': 'warning',
                'feature': 'Arrow Keys',
                'value': arrow_total,
                'message': f'Low arrow key usage ({arrow_total})',
                'detail': 'Limited non-linear editing'
            })
        else:
            indicators.append({
                'type': 'good',
                'feature': 'Arrow Keys',
                'value': arrow_total,
                'message': f'Normal arrow key usage ({arrow_total})',
                'detail': 'Active code navigation observed'
            })
        
        # Deletion ratio
        deletion_ratio = features['deletion_ratio']
        if deletion_ratio < 0.03:
            indicators.append({
                'type': 'critical',
                'feature': 'Deletion Ratio',
                'value': deletion_ratio,
                'message': f'Very low deletion ratio ({deletion_ratio:.1%})',
                'detail': 'Too few corrections - suspiciously perfect'
            })
        elif deletion_ratio < 0.08:
            indicators.append({
                'type': 'warning',
                'feature': 'Deletion Ratio',
                'value': deletion_ratio,
                'message': f'Low deletion ratio ({deletion_ratio:.1%})',
                'detail': 'Fewer corrections than typical'
            })
        else:
            indicators.append({
                'type': 'good',
                'feature': 'Deletion Ratio',
                'value': deletion_ratio,
                'message': f'Normal deletion usage ({deletion_ratio:.1%})',
                'detail': 'Regular corrections observed'
            })
        
        # Run/Submit behavior
        run_count = features.get('run_code_count', 0)
        submit_count = features.get('submit_code_count', 0)
        
        if run_count == 0 and submit_count > 0:
            indicators.append({
                'type': 'critical',
                'feature': 'Testing Behavior',
                'value': run_count,
                'message': 'No test runs before submission',
                'detail': 'Submitted without testing - very confident'
            })
        elif run_count < 2:
            indicators.append({
                'type': 'warning',
                'feature': 'Testing Behavior',
                'value': run_count,
                'message': f'Minimal testing ({run_count} runs)',
                'detail': 'Limited iterative testing'
            })
        else:
            indicators.append({
                'type': 'good',
                'feature': 'Testing Behavior',
                'value': run_count,
                'message': f'Good testing practice ({run_count} runs)',
                'detail': 'Iterative testing observed'
            })
        
        # Edit navigation switches
        edit_switches = features.get('edit_navigation_switches', 0)
        if edit_switches < 5:
            indicators.append({
                'type': 'warning',
                'feature': 'Editing Pattern',
                'value': edit_switches,
                'message': f'Few editing switches ({edit_switches})',
                'detail': 'Linear typing pattern - minimal back-and-forth'
            })
        else:
            indicators.append({
                'type': 'good',
                'feature': 'Editing Pattern',
                'value': edit_switches,
                'message': f'Active editing ({edit_switches} switches)',
                'detail': 'Non-linear editing with back-and-forth navigation'
            })
        
        # Typing variance
        typing_variance = features['typing_variance']
        if typing_variance < 10000:
            indicators.append({
                'type': 'critical',
                'feature': 'Typing Rhythm',
                'value': typing_variance,
                'message': f'Very consistent typing (variance: {typing_variance:.0f})',
                'detail': 'Suspiciously uniform - like copying'
            })
        elif typing_variance < 30000:
            indicators.append({
                'type': 'warning',
                'feature': 'Typing Rhythm',
                'value': typing_variance,
                'message': f'Low typing variance ({typing_variance:.0f})',
                'detail': 'More consistent than typical human typing'
            })
        
        return indicators
    
    def _get_risk_level(self, score):
        """Convert score to risk level"""
        if score < 0.3:
            return 'LOW'
        elif score < 0.6:
            return 'MEDIUM'
        elif score < 0.8:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def format_output(self, result):
        """Format prediction result"""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("🔍 LINEAR TYPING DETECTION REPORT")
        lines.append("=" * 80)
        
        lines.append(f"\n📝 Session ID: {result['session_id']}")
        
        # Verdict
        if result['prediction'] == 'LINEAR_TYPING':
            lines.append(f"\n🚨 VERDICT: LINEAR TYPING DETECTED (Suspicious)")
        else:
            lines.append(f"\n✅ VERDICT: GENUINE PROBLEM-SOLVING")
        
        lines.append(f"📊 Suspicion Score: {result['suspicion_score']:.3f}")
        lines.append(f"⚠️  Risk Level: {result['risk_level']}")
        lines.append(f"💪 Confidence: {result['confidence']:.1%}")
        
        # Risk indicator
        risk_indicators = {
            'CRITICAL': "🔴🔴🔴 CRITICAL - High confidence of cheating",
            'HIGH': "🔴🔴 HIGH RISK - Likely copied from visible source",
            'MEDIUM': "🟡 MEDIUM RISK - Some suspicious patterns",
            'LOW': "🟢 LOW RISK - Normal behavior patterns"
        }
        lines.append(f"\n{risk_indicators[result['risk_level']]}")
        
        # Behavioral stats
        lines.append("\n" + "=" * 80)
        lines.append("📈 BEHAVIORAL STATISTICS")
        lines.append("=" * 80)
        
        stats = result['behavioral_stats']
        lines.append(f"\n  Navigation & Editing:")
        lines.append(f"    • Arrow Keys: {stats['arrow_keys']}")
        lines.append(f"    • Editor Clicks: {stats['editor_clicks']}")
        lines.append(f"    • Edit Switches: {stats['edit_switches']}")
        lines.append(f"    • Modifier Keys: {stats['modifier_usage']}")
        
        lines.append(f"\n  Corrections:")
        lines.append(f"    • Backspaces: {stats['backspaces']}")
        lines.append(f"    • Deletion Ratio: {stats['deletion_ratio']:.1%}")
        
        lines.append(f"\n  Testing:")
        lines.append(f"    • Code Runs: {stats['run_count']}")
        lines.append(f"    • Submissions: {stats['submit_count']}")
        
        lines.append(f"\n  Typing:")
        lines.append(f"    • Typing Variance: {stats['typing_variance']:.0f}")
        
        # Key indicators
        lines.append("\n" + "=" * 80)
        lines.append("🔍 KEY BEHAVIORAL INDICATORS")
        lines.append("=" * 80)
        
        for ind in result['key_indicators']:
            icon = {"critical": "❌", "warning": "⚠️", "good": "✅"}[ind['type']]
            lines.append(f"\n{icon} {ind['feature']}: {ind['message']}")
            lines.append(f"   └─ {ind['detail']}")
        
        # Recommendation
        lines.append("\n" + "=" * 80)
        lines.append("💡 RECOMMENDATION")
        lines.append("=" * 80)
        
        if result['suspicion_score'] > 0.7:
            lines.append("\n🚨 HIGH CONFIDENCE OF CHEATING")
            lines.append("   → Flag for immediate manual review")
            lines.append("   → Consider secondary verification")
        elif result['suspicion_score'] > 0.5:
            lines.append("\n⚠️  MODERATE SUSPICION")
            lines.append("   → Consider additional verification")
            lines.append("   → May warrant closer examination")
        else:
            lines.append("\n✅ LOW SUSPICION - Likely Genuine")
            lines.append("   → Behavioral patterns consistent with authentic solving")
            lines.append("   → Acceptable for automated approval")
        
        lines.append("\n" + "=" * 80 + "\n")
        
        return "\n".join(lines)

def main():
    """Main prediction interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detect linear typing patterns in coding sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('session_file', help='Path to session JSON file')
    parser.add_argument('--model-dir', default='models', help='Model directory')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    if not Path(args.session_file).exists():
        print(f"❌ Error: File not found: {args.session_file}")
        sys.exit(1)
    
    # Load detector
    detector = LinearTypingDetector(model_dir=args.model_dir)
    
    print(f"\n🔍 Analyzing: {args.session_file}")
    
    # Predict
    result = detector.predict_session(args.session_file)
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(detector.format_output(result))

if __name__ == '__main__':
    main()