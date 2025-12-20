"""
Linear Temporal Logic (LTL) Specifications for Drone Control

LTL is fancy formal verification stuff from computer science
We use it to define safety rules and give better reward shaping
Much cleaner than manually tuning reward functions

Fixed circular reference bug in atomic propositions
"""
import numpy as np
from enum import Enum
from collections import deque

class LTLProperty(Enum):
    """Types of temporal properties we can check"""
    SAFETY = "safety"          # □¬bad - "always not bad"
    LIVENESS = "liveness"      # ◊good - "eventually good"
    RESPONSE = "response"      # □(A → ◊B) - "if A then eventually B"
    PERSISTENCE = "persistence"  # ◊□prop - "eventually always"

class AtomicProposition:
    """
    Base building block - a boolean property that's either true or false
    Examples: "at_target", "safe_from_obstacles", etc.
    
    We keep a history so we can check temporal patterns like
    "was true in last N steps" or "has been true for N steps"
    """
    def __init__(self, name, check_fn):
        self.name = name
        self.check_fn = check_fn  # function that takes env and returns bool
        self.history = deque(maxlen=200)  # rolling window
        self._cached_value = None  # avoid re-evaluating same timestep
        self._cached_step = -1
    
    def evaluate(self, env):
        """
        Check if this proposition is true right now
        Uses caching to avoid evaluating multiple times per timestep
        """
        # Cache check - don't re-evaluate if already did this step
        if env.current_step == self._cached_step:
            return self._cached_value
        
        result = self.check_fn(env)
        self.history.append(result)
        self._cached_value = result
        self._cached_step = env.current_step
        return result
    
    def was_true_within(self, steps):
        """Check if proposition was true at any point in last N steps"""
        if len(self.history) < steps:
            return any(self.history)
        return any(list(self.history)[-steps:])
    
    def is_always_true(self, steps):
        """Check if proposition has been continuously true for last N steps"""
        if len(self.history) < steps:
            return False
        return all(list(self.history)[-steps:])
    
    def reset(self):
        """Clear history"""
        self.history.clear()
        self._cached_value = None
        self._cached_step = -1

class LTLSpecification:
    """
    Base class for LTL specifications
    Each spec monitors a temporal property and provides reward shaping
    """
    def __init__(self, name, property_type, description, reward_weight=1.0):
        self.name = name
        self.property_type = property_type
        self.description = description
        self.reward_weight = reward_weight
        self.violations = 0
        self.satisfactions = 0
        self.history = deque(maxlen=1000)
    
    def evaluate(self, env):
        """
        Check specification and return (satisfied, reward_modifier)
        
        Returns:
            satisfied: True/False/None (None if not yet determined)
            reward: float reward modifier
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset spec state between episodes"""
        self.history.clear()

class SafetySpec(LTLSpecification):
    """
    Safety: □¬bad  (always not bad)
    
    Examples:
    - Never collide with obstacles
    - Never go out of bounds
    - Never flip completely over
    
    These are "must never happen" rules
    """
    def __init__(self, name, bad_condition_fn, description, penalty=-100.0):
        super().__init__(name, LTLProperty.SAFETY, description, penalty)
        self.bad_condition_fn = bad_condition_fn
        self.penalty = penalty
    
    def evaluate(self, env):
        """Check if the bad thing happened"""
        is_bad = self.bad_condition_fn(env)
        self.history.append(not is_bad)
        
        if is_bad:
            self.violations += 1
            return False, self.penalty
        else:
            self.satisfactions += 1
            return True, 0.0

class LivenessSpec(LTLSpecification):
    """
    Liveness: ◊good  (eventually good)
    
    Examples:
    - Eventually reach the target
    - Eventually stabilize
    
    These are "must eventually happen" rules
    Has a timeout - if not achieved by then, it's a violation
    """
    def __init__(self, name, good_condition_fn, description, 
                 reward=100.0, timeout_steps=2000):
        super().__init__(name, LTLProperty.LIVENESS, description, reward)
        self.good_condition_fn = good_condition_fn
        self.reward = reward
        self.timeout_steps = timeout_steps
        self.achieved = False
        self.steps_since_reset = 0
    
    def evaluate(self, env):
        """Check if goal achieved or timed out"""
        self.steps_since_reset += 1
        is_good = self.good_condition_fn(env)
        
        if is_good and not self.achieved:
            # Just achieved goal!
            self.achieved = True
            self.satisfactions += 1
            return True, self.reward
        elif not is_good and self.steps_since_reset >= self.timeout_steps:
            # Timed out without achieving goal
            self.violations += 1
            return False, -self.reward * 0.5
        
        return is_good, 0.0
    
    def reset(self):
        super().reset()
        self.achieved = False
        self.steps_since_reset = 0

class ResponseSpec(LTLSpecification):
    """
    Response: □(A → ◊B)  (always: if A happens, then B must eventually happen)
    
    Examples:
    - If enter danger zone → must exit it within N steps
    - If reach target area → must stabilize within N steps
    
    This is for cause-and-effect patterns
    """
    def __init__(self, name, trigger_fn, response_fn, description,
                 response_window=100, penalty=-50.0, reward=20.0):
        super().__init__(name, LTLProperty.RESPONSE, description)
        self.trigger_fn = trigger_fn
        self.response_fn = response_fn
        self.response_window = response_window
        self.penalty = penalty
        self.reward = reward
        
        self.triggered = False
        self.steps_since_trigger = 0
    
    def evaluate(self, env):
        """Check trigger-response pattern"""
        is_triggered = self.trigger_fn(env)
        is_response = self.response_fn(env)
        
        # New trigger detected
        if is_triggered and not self.triggered:
            self.triggered = True
            self.steps_since_trigger = 0
            return None, 0.0
        
        # Currently waiting for response
        if self.triggered:
            self.steps_since_trigger += 1
            
            # Response achieved - good!
            if is_response:
                self.triggered = False
                self.satisfactions += 1
                return True, self.reward
            
            # Timeout - response failed
            elif self.steps_since_trigger >= self.response_window:
                self.triggered = False
                self.violations += 1
                return False, self.penalty
        
        return None, 0.0
    
    def reset(self):
        super().reset()
        self.triggered = False
        self.steps_since_trigger = 0

class PersistenceSpec(LTLSpecification):
    """
    Persistence: ◊□property  (eventually always - once true, stays true)
    
    Examples:
    - Once stable hover is achieved, must maintain it
    - Once safe altitude reached, stay there
    
    This is for "maintain state" requirements
    """
    def __init__(self, name, condition_fn, description,
                 stabilization_steps=50, reward=100.0, penalty=-200.0):
        super().__init__(name, LTLProperty.PERSISTENCE, description)
        self.condition_fn = condition_fn
        self.stabilization_steps = stabilization_steps
        self.reward = reward
        self.penalty = penalty
        
        self.consecutive_true = 0
        self.is_persistent = False
    
    def evaluate(self, env):
        """Check if property becomes and stays true"""
        is_true = self.condition_fn(env)
        
        if is_true:
            self.consecutive_true += 1
            
            # Just achieved persistence!
            if self.consecutive_true == self.stabilization_steps:
                self.is_persistent = True
                self.satisfactions += 1
                return True, self.reward
            
            # Maintaining persistence (small ongoing reward)
            elif self.is_persistent:
                return True, self.reward * 0.1
        
        else:
            # Lost persistence after achieving it - very bad!
            if self.is_persistent:
                self.violations += 1
                self.is_persistent = False
                self.consecutive_true = 0
                return False, self.penalty
            
            # Haven't achieved persistence yet
            self.consecutive_true = 0
        
        return None, 0.0
    
    def reset(self):
        super().reset()
        self.consecutive_true = 0
        self.is_persistent = False

class LTLMonitor:
    """
    Central monitor that tracks all LTL specifications
    Evaluates them each timestep and provides combined reward shaping
    """
    def __init__(self):
        self.specifications = []
        self.atomic_props = {}
    
    def add_atomic_proposition(self, name, check_fn):
        """Register an atomic proposition"""
        self.atomic_props[name] = AtomicProposition(name, check_fn)
    
    def add_specification(self, spec):
        """Register an LTL specification"""
        self.specifications.append(spec)
    
    def evaluate_all(self, env):
        """
        Evaluate all specs and return combined reward
        
        Returns:
            total_reward: sum of all reward modifiers
            violations: list of spec names that were violated
            satisfactions: list of spec names that were satisfied
        """
        # First evaluate all atomic propositions for this timestep
        # This fills their caches so specs can query them efficiently
        for prop in self.atomic_props.values():
            prop.evaluate(env)
        
        # Now evaluate each specification
        total_reward = 0.0
        violations = []
        satisfactions = []
        
        for spec in self.specifications:
            satisfied, reward = spec.evaluate(env)
            
            if satisfied is False:
                violations.append(spec.name)
            elif satisfied is True:
                satisfactions.append(spec.name)
            
            total_reward += reward
        
        return total_reward, violations, satisfactions
    
    def reset_all(self):
        """Reset all specs and propositions for new episode"""
        for spec in self.specifications:
            spec.reset()
        for prop in self.atomic_props.values():
            prop.reset()
    
    def get_statistics(self):
        """Get compliance statistics for all specs"""
        stats = {}
        for spec in self.specifications:
            total = spec.violations + spec.satisfactions
            compliance_rate = (spec.satisfactions / total * 100) if total > 0 else 0
            stats[spec.name] = {
                'type': spec.property_type.value,
                'violations': spec.violations,
                'satisfactions': spec.satisfactions,
                'compliance_rate': compliance_rate
            }
        return stats
    
    def print_statistics(self):
        """Print nice formatted statistics"""
        print("\n" + "="*70)
        print("LTL SPECIFICATION COMPLIANCE STATISTICS")
        print("="*70)
        
        stats = self.get_statistics()
        for name, data in stats.items():
            print(f"\n{name} ({data['type'].upper()})")
            print(f"  Satisfactions: {data['satisfactions']}")
            print(f"  Violations: {data['violations']}")
            print(f"  Compliance Rate: {data['compliance_rate']:.1f}%")
        
        print("="*70)

def create_drone_ltl_monitor(env):
    """
    Create LTL monitor with drone-specific specifications
    
    This sets up all the formal properties we want the drone to satisfy
    Think of it as a formal specification of "correct behavior"
    """
    monitor = LTLMonitor()
    
    # ===== ATOMIC PROPOSITIONS =====
    # These are the basic facts we can check at each timestep
    # We build complex temporal properties from these building blocks
    
    # Position checks
    monitor.add_atomic_proposition(
        "at_target",
        lambda e: np.linalg.norm(e.position - e.target) < e.hover_zone_radius
    )
    
    monitor.add_atomic_proposition(
        "in_bounds",
        lambda e: (np.abs(e.position[0]) < 15 and 
                  np.abs(e.position[1]) < 15 and 
                  e.position[2] > 0.1 and 
                  e.position[2] < 20)
    )
    
    monitor.add_atomic_proposition(
        "safe_from_obstacles",
        lambda e: all(obs.distance_to(e.position) > 0.2 for obs in e.obstacles)
    )
    
    # Motion checks
    monitor.add_atomic_proposition(
        "low_velocity",
        lambda e: np.linalg.norm(e.velocity) < e.max_hover_velocity
    )
    
    monitor.add_atomic_proposition(
        "level_orientation",
        lambda e: (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt
    )
    
    monitor.add_atomic_proposition(
        "stable_angular",
        lambda e: np.linalg.norm(e.angular_velocity) < 0.5
    )
    
    # Composite check - stable hover (used by multiple specs)
    # This one depends on other props, so we evaluate them directly
    monitor.add_atomic_proposition(
        "stable_hover",
        lambda e: (np.linalg.norm(e.position - e.target) < e.hover_zone_radius and
                  np.linalg.norm(e.velocity) < e.max_hover_velocity and
                  (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt and
                  np.linalg.norm(e.angular_velocity) < 0.5)
    )
    
    # ===== SAFETY SPECIFICATIONS (□¬bad) =====
    # These are hard constraints that must NEVER be violated
    
    # Never collide with obstacles
    monitor.add_specification(SafetySpec(
        name="no_collisions",
        bad_condition_fn=lambda e: any(obs.check_collision(e.position) for obs in e.obstacles),
        description="Must never collide with obstacles",
        penalty=-200.0
    ))
    
    # Never go out of bounds
    monitor.add_specification(SafetySpec(
        name="stay_in_bounds",
        bad_condition_fn=lambda e: (np.abs(e.position[0]) > 15 or 
                                   np.abs(e.position[1]) > 15 or 
                                   e.position[2] < 0.1 or 
                                   e.position[2] > 20),
        description="Must stay within designated flight area",
        penalty=-100.0
    ))
    
    # Never flip over completely
    monitor.add_specification(SafetySpec(
        name="no_flip",
        bad_condition_fn=lambda e: (np.abs(e.orientation[0]) > np.pi*0.8 or 
                                   np.abs(e.orientation[1]) > np.pi*0.8),
        description="Must not flip beyond 144 degrees (gimbal lock territory)",
        penalty=-80.0
    ))
    
    # ===== LIVENESS SPECIFICATIONS (◊good) =====
    # These are goals that must eventually be achieved
    
    # Eventually reach the target area
    monitor.add_specification(LivenessSpec(
        name="reach_target",
        good_condition_fn=lambda e: np.linalg.norm(e.position - e.target) < e.hover_zone_radius,
        description="Must eventually reach target zone",
        reward=50.0,
        timeout_steps=2000
    ))
    
    # Eventually achieve stable flight
    monitor.add_specification(LivenessSpec(
        name="stabilize",
        good_condition_fn=lambda e: (np.linalg.norm(e.velocity) < e.max_hover_velocity and
                                    (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt),
        description="Must eventually achieve stable controlled flight",
        reward=30.0,
        timeout_steps=1500
    ))
    
    # ===== RESPONSE SPECIFICATIONS (□(A → ◊B)) =====
    # If-then temporal rules
    
    # If near obstacle, must move away
    monitor.add_specification(ResponseSpec(
        name="avoid_danger",
        trigger_fn=lambda e: any(obs.distance_to(e.position) < 0.5 for obs in e.obstacles),
        response_fn=lambda e: all(obs.distance_to(e.position) > 1.0 for obs in e.obstacles),
        description="If too close to obstacle, must move to safe distance",
        response_window=80,
        penalty=-40.0,
        reward=15.0
    ))
    
    # If near target, must stabilize
    monitor.add_specification(ResponseSpec(
        name="stabilize_at_target",
        trigger_fn=lambda e: np.linalg.norm(e.position - e.target) < e.hover_zone_radius * 1.5,
        response_fn=lambda e: (np.linalg.norm(e.velocity) < e.max_hover_velocity and
                              (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt),
        description="Once near target, must quickly stabilize",
        response_window=150,
        penalty=-30.0,
        reward=25.0
    ))
    
    # ===== PERSISTENCE SPECIFICATIONS (◊□property) =====
    # Once achieved, must be maintained
    
    # Once stable hover is achieved, maintain it
    # This is the ultimate goal - get there and stay there
    monitor.add_specification(PersistenceSpec(
        name="maintain_hover",
        condition_fn=lambda e: (np.linalg.norm(e.position - e.target) < e.hover_zone_radius and
                               np.linalg.norm(e.velocity) < e.max_hover_velocity and
                               (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt and
                               np.linalg.norm(e.angular_velocity) < 0.5),
        description="Once stable hover achieved, must maintain it persistently",
        stabilization_steps=env.hover_time_required,
        reward=300.0,
        penalty=-250.0
    ))
    
    return monitor