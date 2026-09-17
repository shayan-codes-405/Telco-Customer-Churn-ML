import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Union
from pathlib import Path
from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_NAMES,
    AREA_CODE_MAP,
    BINARY_MAP
)


class ChurnPredictor:
    """
    Inference and Diagnostic Engine for Telecom Customer Churn Prediction.
    Ensures correct categorical mapping, scaling, and risk analysis.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ChurnPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH, scaler_path: Union[str, Path] = SCALER_PATH):
        if self._initialized:
            return
        
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found at: {self.scaler_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self._initialized = True

    def _clean_and_map_value(self, val: Any, mapping: Dict[Any, int], default: int = 0) -> int:
        if val in mapping:
            return mapping[val]
        if isinstance(val, str):
            clean_str = val.strip().lower()
            for k, v in mapping.items():
                if str(k).lower() == clean_str:
                    return v
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def prepare_dataframe(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        """
        Validates, maps categorical fields, and formats into standard feature DataFrame.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Input data must be a dictionary or a pandas DataFrame.")

        # Match columns case-insensitively if needed
        col_map = {col.lower().strip(): col for col in df.columns}
        
        processed_data = {}
        for feature in FEATURE_NAMES:
            actual_col = col_map.get(feature.lower())
            if actual_col is not None:
                val = df[actual_col]
            elif feature in df:
                val = df[feature]
            else:
                raise ValueError(f"Missing required feature column: '{feature}'")

            # Apply specific mappings
            if feature == "area_code":
                processed_data[feature] = val.apply(lambda x: self._clean_and_map_value(x, AREA_CODE_MAP, default=1))
            elif feature in ("international_plan", "voice_mail_plan"):
                processed_data[feature] = val.apply(lambda x: self._clean_and_map_value(x, BINARY_MAP, default=0))
            else:
                processed_data[feature] = pd.to_numeric(val, errors='coerce').fillna(0.0)

        feature_df = pd.DataFrame(processed_data, columns=FEATURE_NAMES)
        return feature_df

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs single customer diagnostic with risk tier, risk drivers, and retention advice.
        """
        input_df = self.prepare_dataframe(customer_data)
        scaled_input = self.scaler.transform(input_df)
        
        prediction = int(self.model.predict(scaled_input)[0])
        prob_churn = float(self.model.predict_proba(scaled_input)[0][1])
        
        # Risk Tier Classification
        if prob_churn >= 0.65:
            risk_tier = "HIGH CHURN RISK"
            risk_badge = "Critical"
            risk_color = "#ef4444"
            summary_text = "Action Needed: Customer displays severe churn signals and is very likely to cancel soon."
        elif prob_churn >= 0.35:
            risk_tier = "MEDIUM CHURN RISK"
            risk_badge = "Warning"
            risk_color = "#f59e0b"
            summary_text = "Warning Sign: Customer shows moderate dissatisfaction signals. Proactive retention recommended."
        else:
            risk_tier = "LOW CHURN RISK"
            risk_badge = "Safe"
            risk_color = "#10b981"
            summary_text = "Safe Account: Customer is stable, satisfied, and highly likely to continue subscription."

        # Compute Explainability Risk Drivers
        service_calls = float(customer_data.get("number_customer_service_calls", 0))
        intl_plan = self._clean_and_map_value(customer_data.get("international_plan", 0), BINARY_MAP)
        total_intl_charge = float(customer_data.get("total_intl_charge", 0.0))
        day_charge = float(customer_data.get("total_day_charge", 0.0))
        tenure = float(customer_data.get("account_length", 0))

        risk_drivers = []
        if service_calls >= 4:
            risk_drivers.append({
                "severity": "high",
                "icon": "🎧",
                "title": f"High Customer Support Calls ({int(service_calls)} calls)",
                "description": "Customer has reached support 4+ times recently, indicating unresolved technical or billing dissatisfaction."
            })
        elif service_calls >= 2:
            risk_drivers.append({
                "severity": "medium",
                "icon": "⚠️",
                "title": f"Elevated Support Escalation ({int(service_calls)} calls)",
                "description": "Customer has required support assistance multiple times in the last billing cycle."
            })

        if intl_plan == 1 and total_intl_charge > 3.0:
            risk_drivers.append({
                "severity": "high",
                "icon": "✈️",
                "title": f"High International Spending (${total_intl_charge:.2f})",
                "description": "Customer pays substantial out-of-plan international fees, which is a major driver of bill shock and carrier switching."
            })

        if day_charge > 40.0:
            risk_drivers.append({
                "severity": "medium",
                "icon": "💰",
                "title": f"High Daytime Peak Usage (${day_charge:.2f}/mo)",
                "description": "Above-average daytime call volume leads to higher recurring monthly billing sensitivity."
            })

        if tenure < 12:
            risk_drivers.append({
                "severity": "medium",
                "icon": "⏳",
                "title": f"Early Lifecycle Account ({int(tenure)} months)",
                "description": "First-year subscribers have lower switching barriers and are more receptive to competitor onboarding promotions."
            })

        if not risk_drivers:
            risk_drivers.append({
                "severity": "low",
                "icon": "💚",
                "title": "Healthy Account Pattern",
                "description": "Low customer care friction, predictable monthly charges, and stable tenure history."
            })

        # Retention Recommendations
        retention_actions = []
        if service_calls >= 3:
            retention_actions.append({
                "priority": "P1 - Immediate",
                "title": "Assign Dedicated Care Specialist",
                "action": "Schedule a direct callback from a senior supervisor to resolve any open tickets within 24 hours."
            })
        if intl_plan == 1 and total_intl_charge > 3.0:
            retention_actions.append({
                "priority": "P2 - Commercial",
                "title": "Apply 20% International Rate Discount",
                "action": "Provide a 6-month promotional discount on international rates to lower average bill cost."
            })
        if day_charge > 40.0:
            retention_actions.append({
                "priority": "P2 - Plan Upgrade",
                "title": "Upgrade to Unlimited Daytime Bundle",
                "action": "Offer a flat-rate unlimited daytime calling tier to eliminate bill volatility."
            })
        if tenure < 12 and prob_churn > 0.40:
            retention_actions.append({
                "priority": "P3 - Loyalty",
                "title": "New Subscriber Courtesy Credit",
                "action": "Apply a $15 anniversary loyalty bill credit on their upcoming billing cycle."
            })
        if not retention_actions:
            retention_actions.append({
                "priority": "Routine",
                "title": "Maintain Standard Loyalty Engagement",
                "action": "No urgent intervention needed. Include in regular quarterly satisfaction surveys."
            })

        return {
            "predicted_churn": prediction,
            "churn_probability": round(prob_churn, 4),
            "churn_probability_pct": f"{prob_churn * 100:.1f}%",
            "risk_tier": risk_tier,
            "risk_badge": risk_badge,
            "risk_color": risk_color,
            "summary_text": summary_text,
            "risk_drivers": risk_drivers,
            "retention_actions": retention_actions,
            "input_features": customer_data
        }

    def predict_batch(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs batch scoring for uploaded CSV datasets with executive analytics summary.
        """
        feature_df = self.prepare_dataframe(df)
        scaled_input = self.scaler.transform(feature_df)
        
        preds = self.model.predict(scaled_input)
        probs = self.model.predict_proba(scaled_input)[:, 1]
        
        result_df = df.copy()
        result_df["predicted_churn"] = preds
        result_df["churn_probability"] = np.round(probs, 4)
        result_df["churn_probability_pct"] = [f"{p * 100:.1f}%" for p in probs]
        
        risk_tiers = []
        for p in probs:
            if p >= 0.65:
                risk_tiers.append("HIGH CHURN RISK")
            elif p >= 0.35:
                risk_tiers.append("MEDIUM CHURN RISK")
            else:
                risk_tiers.append("LOW CHURN RISK")
        result_df["risk_tier"] = risk_tiers

        total = len(result_df)
        churn_count = int(np.sum(preds == 1))
        high_risk_count = int(np.sum(probs >= 0.65))
        med_risk_count = int(np.sum((probs >= 0.35) & (probs < 0.65)))
        low_risk_count = int(np.sum(probs < 0.35))
        
        summary_stats = {
            "total_records": total,
            "predicted_churn_count": churn_count,
            "predicted_retain_count": total - churn_count,
            "churn_rate_pct": f"{(churn_count / total * 100):.1f}%" if total > 0 else "0%",
            "high_risk_count": high_risk_count,
            "medium_risk_count": med_risk_count,
            "low_risk_count": low_risk_count,
            "high_risk_pct": f"{(high_risk_count / total * 100):.1f}%" if total > 0 else "0%",
            "avg_churn_probability": f"{(np.mean(probs) * 100):.1f}%" if total > 0 else "0%"
        }
        
        return result_df, summary_stats
