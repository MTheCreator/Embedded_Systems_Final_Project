"""
Linear Temporal Logic (LTL) Specifications for Drone Control
Defines safety properties, liveness properties, and reward shaping based on LTL
"""
import numpy as np
from enum import Enum
from collections import deque

class LTLProperty(Enum):
    """Types of LTL properties"""
    SAFETY = "safety"      # Something bad never happens (□¬bad)
    LIVENESS = "liveness"  # Something good eventually happens (◊good)
    RESPONSE = "response"  # If A happens, B must eventually follow (□(A → ◊B))
    PERSISTENCE = "persistence"  # Once true, stays true (◊□property)

class AtomicProposition:
    """Atomic propositions that can be true/false at each timestep"""
    def __init__(self, name, check_fn):
        self.name = name
        self.check_fn = check_fn
        self.history = deque(maxlen=200)
    
    def evaluate(self, env):
        """Evaluate if this proposition is true given environment state"""
        result = self.check_fn(env)
        self.history.append(result)
        return result
    
    def was_true_within(self, steps):
        """Check if proposition was true in last N steps"""
        if len(self.history) < steps:
            return any(self.history)
        return any(list(self.history)[-steps:])
    
    def is_always_true(self, steps):
        """Check if proposition has been true for last N steps"""
        if len(self.history) < steps:
            return False
        return all(list(self.history)[-steps:])

class LTLSpecification:
    """
    Represents an LTL specification with reward shaping
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
        """Evaluate specification and return (satisfied, reward_modifier)"""
        raise NotImplementedError
    
    def reset(self):
        """Reset specification state"""
        self.history.clear()

class SafetySpec(LTLSpecification):
    """
    Safety specification: □¬bad (always not bad)
    Example: Always avoid collisions, always stay in bounds
    """
    def __init__(self, name, bad_condition_fn, description, penalty=-100.0):
        super().__init__(name, LTLProperty.SAFETY, description, penalty)
        self.bad_condition_fn = bad_condition_fn
        self.penalty = penalty
    
    def evaluate(self, env):
        """Check if bad condition occurred"""
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
    Liveness specification: ◊good (eventually good)
    Example: Eventually reach target, eventually stabilize
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
        """Check if good condition achieved"""
        self.steps_since_reset += 1
        is_good = self.good_condition_fn(env)
        
        if is_good and not self.achieved:
            self.achieved = True
            self.satisfactions += 1
            return True, self.reward
        elif not is_good and self.steps_since_reset >= self.timeout_steps:
            self.violations += 1
            return False, -self.reward * 0.5
        
        return is_good, 0.0
    
    def reset(self):
        super().reset()
        self.achieved = False
        self.steps_since_reset = 0

class ResponseSpec(LTLSpecification):
    """
    Response specification: □(A → ◊B)
    "Always, if A happens, then B must eventually happen"
    Example: If enter danger zone, must exit it within N steps
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
        
        # New trigger
        if is_triggered and not self.triggered:
            self.triggered = True
            self.steps_since_trigger = 0
            return None, 0.0
        
        # Waiting for response
        if self.triggered:
            self.steps_since_trigger += 1
            
            # Response achieved!
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
    Persistence specification: ◊□property
    "Eventually always" - once achieved, must stay true
    Example: Once stable hover achieved, must maintain it
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
            
            # Maintaining persistence
            elif self.is_persistent:
                return True, self.reward * 0.1  # Small reward for maintaining
        
        else:
            # Lost persistence - big penalty!
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
    Monitors multiple LTL specifications and provides reward shaping
    """
    def __init__(self):
        self.specifications = []
        self.atomic_props = {}
    
    def add_atomic_proposition(self, name, check_fn):
        """Add an atomic proposition"""
        self.atomic_props[name] = AtomicProposition(name, check_fn)
    
    def add_specification(self, spec):
        """Add an LTL specification"""
        self.specifications.append(spec)
    
    def evaluate_all(self, env):
        """Evaluate all specifications and return total reward modifier"""
        # First evaluate atomic propositions
        for prop in self.atomic_props.values():
            prop.evaluate(env)
        
        # Then evaluate specifications
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
        """Reset all specifications"""
        for spec in self.specifications:
            spec.reset()
        for prop in self.atomic_props.values():
            prop.history.clear()
    
    def get_statistics(self):
        """Get statistics about specification compliance"""
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
        """Print statistics in readable format"""
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
    """
    monitor = LTLMonitor()
    
    # ===== ATOMIC PROPOSITIONS =====
    
    monitor.add_atomic_proposition(
        "at_target",
        lambda e: np.linalg.norm(e.position - e.target) < e.hover_zone_radius
    )
    
    monitor.add_atomic_proposition(
        "low_velocity",
        lambda e: np.linalg.norm(e.velocity) < e.max_hover_velocity
    )
    
    monitor.add_atomic_proposition(
        "level_orientation",
        lambda e: (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt
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
    
    monitor.add_atomic_proposition(
        "stable",
        lambda e: (monitor.atomic_props["at_target"].evaluate(e) and
                  monitor.atomic_props["low_velocity"].evaluate(e) and
                  monitor.atomic_props["level_orientation"].evaluate(e))
    )
    
    # ===== SAFETY SPECIFICATIONS (□¬bad) =====
    
    # Never collide with obstacles
    monitor.add_specification(SafetySpec(
        name="no_collisions",
        bad_condition_fn=lambda e: any(obs.check_collision(e.position) for obs in e.obstacles),
        description="Always avoid collisions with obstacles",
        penalty=-200.0
    ))
    
    # Never go out of bounds
    monitor.add_specification(SafetySpec(
        name="stay_in_bounds",
        bad_condition_fn=lambda e: (np.abs(e.position[0]) > 15 or 
                                   np.abs(e.position[1]) > 15 or 
                                   e.position[2] < 0.1 or 
                                   e.position[2] > 20),
        description="Always stay within flight boundaries",
        penalty=-100.0
    ))
    
    # Never flip over completely
    monitor.add_specification(SafetySpec(
        name="no_flip",
        bad_condition_fn=lambda e: (np.abs(e.orientation[0]) > np.pi*0.8 or 
                                   np.abs(e.orientation[1]) > np.pi*0.8),
        description="Never flip over beyond 144 degrees",
        penalty=-80.0
    ))
    
    # ===== LIVENESS SPECIFICATIONS (◊good) =====
    
    # Eventually reach target
    monitor.add_specification(LivenessSpec(
        name="reach_target",
        good_condition_fn=lambda e: np.linalg.norm(e.position - e.target) < e.hover_zone_radius,
        description="Eventually reach the target zone",
        reward=50.0,
        timeout_steps=2000
    ))
    
    # Eventually stabilize
    monitor.add_specification(LivenessSpec(
        name="stabilize",
        good_condition_fn=lambda e: (np.linalg.norm(e.velocity) < e.max_hover_velocity and
                                    (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt),
        description="Eventually achieve stable flight",
        reward=30.0,
        timeout_steps=1500
    ))
    
    # ===== RESPONSE SPECIFICATIONS (□(A → ◊B)) =====
    
    # If too close to obstacle, must move away
    monitor.add_specification(ResponseSpec(
        name="avoid_danger",
        trigger_fn=lambda e: any(obs.distance_to(e.position) < 0.5 for obs in e.obstacles),
        response_fn=lambda e: all(obs.distance_to(e.position) > 1.0 for obs in e.obstacles),
        description="If near obstacle, must move to safe distance",
        response_window=80,
        penalty=-40.0,
        reward=15.0
    ))
    
    # If reach target, must stabilize quickly
    monitor.add_specification(ResponseSpec(
        name="stabilize_at_target",
        trigger_fn=lambda e: np.linalg.norm(e.position - e.target) < e.hover_zone_radius * 1.5,
        response_fn=lambda e: (np.linalg.norm(e.velocity) < e.max_hover_velocity and
                              (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt),
        description="Once near target, must stabilize within window",
        response_window=150,
        penalty=-30.0,
        reward=25.0
    ))
    
    # ===== PERSISTENCE SPECIFICATIONS (◊□property) =====
    
    # Once stable hover achieved, maintain it
    # ===== PERSISTENCE SPECIFICATIONS (◊□property) =====

    # Once stable hover achieved, maintain it
    monitor.add_specification(PersistenceSpec(
        name="maintain_hover",
        condition_fn=lambda e: (np.linalg.norm(e.position - e.target) < e.hover_zone_radius and
                            np.linalg.norm(e.velocity) < e.max_hover_velocity and
                            (np.abs(e.orientation[0]) + np.abs(e.orientation[1])) < e.max_hover_tilt),
        description="Once stable hover achieved, maintain it persistently",
        stabilization_steps=env.hover_time_required,  # ✅ FIXED - use 'env' not 'e'
        reward=300.0,
        penalty=-250.0
    ))
        
    return monitor