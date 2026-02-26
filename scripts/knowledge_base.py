# -*- coding: utf-8 -*-

from typing import Dict, Any


class KnowledgeBase:
    
    def __init__(self):
        self.prior_knowledge: Dict[str, Any] = {}
        self._initialize_dataset_knowledge()
    
    def _initialize_dataset_knowledge(self):
        self.prior_knowledge["localization_agent_TODS"] = """
## Domain Knowledge:
For TODS dataset, it is a very smooth sine wave. There may more than one dip or spike. Detect those deviate the smooth wave pattern as anomaly points **as much as possible**.

Depart from point-level anomaly, period-level anomaly also exist that does not follow the variation trend of a normal sine wave, exhibiting sudden accelerated upward movements, abrupt trend reversals, frequency changes (normal period is 100)
For period-level analysis, you can only analyze in the first global picture!!.

"""
        
        self.prior_knowledge["localization_agent_YAHOO"] = """
## Domain Knowledge:
For YAHOO,For dip and spike, only flag those without reapting with same interval. Also consider those spike caused by noise.
Not each period has anomaly point/period, which is extremely rare.
"""
        
        self.prior_knowledge["planning_agent_TODS"] = """
## Domain Knowledge:
This is TODS dataset.
- TODS dataset is a sine wave series with 100 point as a cycle.
- It is smooth in normal pattern but with few large deviation. Try use global diff Z-score(set threshold about 2) and local-z-score(threshold for 2.5).
- value-based z score may not useful because it is not stable.
- Put it in plan: If two adjacent or nearby points have opposite and large diff Z-scores, check the points between them for significant deviation from surrounding values. If present, flag those intermediate points as anomalies, but exclude the last recovery point.
"""
        
        self.prior_knowledge["planning_agent_YAHOO"] = """
## Domain Knowledge:
- The time series below is from YAHOO, which consist only of point-wise outlier instead of long regime shift period.
Only guide action agent to detect some period that have extreme spike, instead of some periods themselves.
- For time series in YAHOO, The shift in the overall pattern (regime change) should not be treated as an anomaly itself. 
Instead, anomalies should only be detected as abrupt deviations that occur within a single, consistent regime.
- Note: Use local_diff_zscore() and global_diff_zscore(). set a threshold 3.5. DO NOT call fit_periodic_pattern().
- Try to minilize to interval to those truely anomaly, instead of theri neighbor.
- If two adjacent or nearby points have opposite and large diff Z-scores, check the points between them for significant deviation from surrounding values. If present, flag those intermediate points as anomalies, but exclude the last recovery point.
"""
        
        self.prior_knowledge["localization_agent_IOPS"] = """
## Domain Knowledge:
For this,For dip and spike, only flag those extremely strange one. Most spike and dip is normal noise!
Not each period has anomaly point/period, which is extremely rare.
"""
        
        self.prior_knowledge["planning_agent_IOPS"] = """
## Domain Knowledge:
- The time series you detect is IOPS dataset
- For IOPS dataset, it might be noise, with rare anomaly. 
- For the IOPS dataset, only those qualify as anomalies: 
    * Isolated abrupt point
    * Small clusters of points that form spikes, deep dips, or sudden steps (high/low platforms)
- Slow upward or downward trends, as well as gradual peaks and valleys caused by such trends, are completely normal behavior and must NOT be flagged as anomalies.
- TOOLCALL: For IOPS dataset, You can call Gloabl Z score( threshold~3) and Global diff Z-score (Threshold~3.5), once any of two detect, label it as anomaly.
- IMPORTANT: YOU NEED TO PUT THIS INTO YOUR PLAN: When detect spike, try to label it's left two and right two also as anomaly(Total 5).
"""
        
        self.prior_knowledge["localization_agent_ECG"] = """
## Domain Knowledge:
This is ECG dataset, a highly periodic and regular time series signal, characteristic of bio-signals like the Electrocardiogram (ECG).
Waveform Morphology (Shape): Each cycle is dominated by a single, high-amplitude, sharp spike which represents the main electrical event (analogous to the QRS complex in ECG).
However, one cycle may longer than the boundingbox.
So you should only refer to the first global picture.
So do not report any local anomaly about dip or spike or Level Shift/Drift.
"""
        
        self.prior_knowledge["planning_agent_ECG"] = """
## Domain Knowledge:
The time series is from ECG, a highly periodic Electrocardiogram dataset. 
Important:
**Only use fit_periodic_pattern() to detect anomaly**
**If fit_periodic_pattern() return is_anomaly: true, flag the **whole 100 period as anomaly**. Put this into your plan**.
**Else, do not report any anomaly**
"""
        
        self.prior_knowledge["localization_agent_SVDB"] = """
## Domain Knowledge:
This series contain much noise. Do not report unless **it's value is far more or less than average**
"""
        
        self.prior_knowledge["planning_agent_SVDB"] = """
## Domain Knowledge:
This series is from SVDB dataset.
This series contain much noise with a ECG record.
calling fit_periodic_pattern() to detect those long-periodic anomaly.
Put it in your plan: If detect the spike or dip, label the **whole series(100 point)** as anomaly.
"""
        
        self.prior_knowledge["localization_agent_NAB"] = """
## Domain Knowledge:
This series contain much noise. Do not report unless **it's value is far more or less than average**
"""
        
        self.prior_knowledge["planning_agent_NAB"] = """
## Domain Knowledge:
This series contain much noise.
Never calling fit_periodic_pattern(). Use global_value_zscore() to detect those global deviations. Set threshold more than 5.
Put it in your plan: If detect the spike or dip, label the **whole series(100 point)** as anomaly.
"""
        
        self.prior_knowledge["fine_grained_agent_TODS"] = """
## Domain Knowledge for Fine-Grained Reasoning:
The Dataset is from **TODS**, include a few of anomaly point and period. 
You should conduct based on **TODS** domain knowledge.
- Single-point anomalies exhibiting a sharp 'spike-and-reversal' pattern in Z-score differentials
- Short-duration cluster anomalies characterized by a persistent Z-score deviation, followed by a multi-step recovery phase..
"""
        
        self.prior_knowledge["fine_grained_agent_YAHOO"] = """
## Domain Knowledge for Fine-Grained Reasoning:
The Dataset is from YAHOO, include rare local and global anomaly point.
Try to minilize to interval to those truely anomaly, instead of their neighbor.
If two adjacent or nearby points have opposite and large diff Z-scores, check the points between them for significant deviation from surrounding values. If present, flag **those intermediate points as anomalies**, but exclude the last recovery point.
eg. Include sustained high plateau after abrupt spike onset, but do not label the last recovery one.
The Dataset is from **YAHOO**, include a few of anomaly point and period. 
You should conduct based on **YAHOO** domain knowledge.
"""
        
        self.prior_knowledge["fine_grained_agent_IOPS"] = """
## Domain Knowledge for Fine-Grained Reasoning:
The Dataset is from IOPS, include local and global anomaly point.
When detect spike, try to label it's left two and right two also as anomaly(Total 5).
The Dataset is from **IOPS**, include a few of anomaly point and period. 
You should conduct based on **IOPS** domain knowledge.
"""
    
        
        self.prior_knowledge["localization_agent_WSD"] = """
## Domain Knowledge:
"""
        
        self.prior_knowledge["planning_agent_WSD"] = """
## Domain Knowledge:
There may rare but extremely abrupt anomaly one or few-point spike/dip.
Try to use global diff Z-score(threshold about 4) to detect those abrupt rise and fall and global value Z-score(threshold about 3.5) 
If the diff Z-score or value Z-score show anomaly, try to scan and catch the all interval.
to detect those global higher or deeper cluster and spike.
The series is may change over and over again, so local-window-based z-score method will not useful.
Try to label the whole anomaly period (The anomaly usually be a interval instead one point).
(You may detect the rise/fall and recovery position by global-diff-Z-score, and they hold opposite sign)
"""
        
        self.prior_knowledge["fine_grained_agent_WSD"] = """
## Domain Knowledge for Fine-Grained Reasoning:
The Dataset is from **WSD**, include a few of anomaly point and period. 
You should conduct based on **WSD** domain knowledge.
There are rare few-point extremely spike/dip/cluster in WSD.
Try to label the whole anomaly period (The anomaly usually be a interval instead one point).
(You may detect the rise/fall and recovery position by global-diff-Z-score or the change of value, and they hold opposite sign)
"""
    
    def detect_dataset_type(self, folder_path: str = None, sample_name: str = None) -> str:
        text_to_check = ""
        if folder_path:
            text_to_check += folder_path + " "
        if sample_name:
            text_to_check += sample_name + " "
        
        text_to_check = text_to_check.upper()
        
        if "TODS" in text_to_check:
            return "TODS"
        elif "YAHOO" in text_to_check:
            return "YAHOO"
        elif "IOPS" in text_to_check:
            return "IOPS"
        elif "WSD" in text_to_check:
            return "WSD"
        else:
            return "UNKNOWN"
    
    def get_agent_knowledge(self, agent_type: str, dataset_type: str = None) -> str:
        # locator 使用 planning_agent 的 key，detector 使用 fine_grained_agent 的 key
        lookup_agent = "planning_agent" if agent_type == "locator" else "fine_grained_agent" if agent_type == "detector" else agent_type
        if dataset_type:
            dataset_type = dataset_type.upper()
            key = f"{lookup_agent}_{dataset_type}"
            knowledge = self.prior_knowledge.get(key, "")
            if knowledge:
                return knowledge
        

        if agent_type == "localization_agent":
            tods_knowledge = self.prior_knowledge.get("localization_agent_TODS", "")
            yahoo_knowledge = self.prior_knowledge.get("localization_agent_YAHOO", "")
            iops_knowledge = self.prior_knowledge.get("localization_agent_IOPS", "")
            wsd_knowledge = self.prior_knowledge.get("localization_agent_WSD", "")
            return tods_knowledge + "\n" + yahoo_knowledge + "\n" + iops_knowledge + "\n"  + wsd_knowledge
        elif agent_type in ("planning_agent", "locator"):
            tods_knowledge = self.prior_knowledge.get("planning_agent_TODS", "")
            yahoo_knowledge = self.prior_knowledge.get("planning_agent_YAHOO", "")
            iops_knowledge = self.prior_knowledge.get("planning_agent_IOPS", "")
            wsd_knowledge = self.prior_knowledge.get("planning_agent_WSD", "")
            return tods_knowledge + "\n" + yahoo_knowledge + "\n" + iops_knowledge + "\n" + wsd_knowledge
        elif agent_type in ("fine_grained_agent", "detector"):
            tods_knowledge = self.prior_knowledge.get("fine_grained_agent_TODS", "")
            yahoo_knowledge = self.prior_knowledge.get("fine_grained_agent_YAHOO", "")
            iops_knowledge = self.prior_knowledge.get("fine_grained_agent_IOPS", "")
            wsd_knowledge = self.prior_knowledge.get("fine_grained_agent_WSD", "")
            return tods_knowledge + "\n" + yahoo_knowledge + "\n" + iops_knowledge + "\n" + wsd_knowledge
        return ""

