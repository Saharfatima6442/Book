---
sidebar_position: 11
---

# Chapter 11: Safety and Ethics in Humanoid Robotics

## Introduction to Safety in Humanoid Robotics

Safety in humanoid robotics encompasses both physical safety for humans and robots, as well as operational reliability. Unlike industrial robots operating in controlled environments, humanoid robots interact directly with humans in unstructured spaces, making safety considerations paramount for their acceptance and deployment.

### Safety vs. Security in Robotics

**Safety** refers to preventing harm from normal operation and foreseeable misuse:
- Physical harm prevention
- Operational reliability
- Failure mode management

**Security** addresses protection against malicious attacks:
- Cybersecurity measures
- Authentication and authorization
- Data protection

Both aspects are critical for humanoid robots operating in human environments.

## Physical Safety Systems

### Collision Avoidance and Force Limiting

Humanoid robots must prevent harmful collisions and limit interaction forces:

#### Force Control and Compliance
```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class SafetyController:
    def __init__(self):
        self.max_force_threshold = 50.0  # Newtons
        self.max_torque_threshold = 10.0  # Newton-meters
        self.collision_buffer = 0.5  # meters
        self.emergency_stop_active = False

    def check_force_limits(self, joint_torques, cartesian_forces):
        """
        Check if forces exceed safety thresholds
        """
        max_torque_exceeded = any(
            abs(torque) > self.max_torque_threshold for torque in joint_torques
        )

        max_force_exceeded = any(
            np.linalg.norm(force) > self.max_force_threshold for force in cartesian_forces
        )

        if max_torque_exceeded or max_force_exceeded:
            self.trigger_safety_stop()
            return False

        return True

    def calculate_compliant_motion(self, desired_pose, external_force, stiffness=1000):
        """
        Calculate compliant motion to reduce interaction forces
        """
        # Implement admittance control for compliance
        compliance_matrix = np.eye(6) / stiffness  # 6-DOF compliance
        force_correction = compliance_matrix @ external_force
        compliant_pose = desired_pose + force_correction

        return compliant_pose

    def trigger_safety_stop(self):
        """
        Activate emergency stop procedures
        """
        self.emergency_stop_active = True
        self.apply_brakes()
        self.log_safety_event("Force limit exceeded", "Emergency stop activated")

    def apply_brakes(self):
        """
        Apply emergency brakes to all joints
        """
        # Send emergency stop commands to all joint controllers
        pass
```

### Collision Detection and Avoidance

Advanced collision detection for humanoid robots:

#### Real-Time Collision Detection
```python
import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d

class CollisionDetector:
    def __init__(self, robot_model, safety_margin=0.1):
        self.robot_model = robot_model  # URDF or similar model
        self.safety_margin = safety_margin
        self.human_detection_threshold = 0.8  # meters

    def detect_collisions(self, robot_pose, environment_point_cloud, humans=None):
        """
        Detect potential collisions with environment and humans
        """
        # Get robot's current configuration
        robot_points = self.robot_model.get_surface_points(robot_pose)

        collision_risk = {
            'environment': False,
            'humans': False,
            'self_collision': False,
            'details': []
        }

        # Check environment collisions
        if environment_point_cloud:
            env_collision = self.check_environment_collision(
                robot_points, environment_point_cloud
            )
            collision_risk['environment'] = env_collision['risk']
            collision_risk['details'].extend(env_collision['details'])

        # Check human collisions
        if humans:
            human_collision = self.check_human_collision(robot_points, humans)
            collision_risk['humans'] = human_collision['risk']
            collision_risk['details'].extend(human_collision['details'])

        # Check self-collisions
        self_collision = self.check_self_collision(robot_pose)
        collision_risk['self_collision'] = self_collision['risk']
        collision_risk['details'].extend(self_collision['details'])

        return collision_risk

    def check_environment_collision(self, robot_points, env_points):
        """
        Check for collisions with environment obstacles
        """
        # Calculate minimum distances
        distances = cdist(robot_points, env_points)
        min_distances = np.min(distances, axis=1)

        collision_risk = np.any(min_distances < self.safety_margin)
        collision_details = [
            {'type': 'environment', 'distance': d}
            for d in min_distances if d < self.safety_margin
        ]

        return {
            'risk': collision_risk,
            'details': collision_details
        }

    def check_human_collision(self, robot_points, humans):
        """
        Check for collisions with humans
        """
        collision_risk = False
        collision_details = []

        for human in humans:
            human_points = self.estimate_human_shape(human)
            distances = cdist(robot_points, human_points)
            min_distance = np.min(distances)

            if min_distance < self.human_detection_threshold:
                collision_risk = True
                collision_details.append({
                    'type': 'human',
                    'human_id': human.get('id', 'unknown'),
                    'distance': min_distance
                })

        return {
            'risk': collision_risk,
            'details': collision_details
        }

    def estimate_human_shape(self, human_data):
        """
        Estimate human body shape from detection data
        """
        # Simplified human model as bounding box or capsule
        # In practice, this would use pose estimation data
        pass
```

### Safe Human-Robot Interaction Protocols

Protocols for safe interaction between humans and robots:

#### Proxemics-Based Safety
```python
class ProxemicsSafety:
    def __init__(self):
        # Personal space zones (distances in meters)
        self.intimate_zone = 0.45    # 0-1.5 feet
        self.personal_zone = 1.2     # 1.5-4 feet
        self.social_zone = 3.6       # 4-12 feet
        self.public_zone = 7.6       # 12+ feet

    def assess_interaction_safety(self, robot_position, human_position):
        """
        Assess safety based on proxemics principles
        """
        distance = np.linalg.norm(robot_position - human_position)

        safety_assessment = {
            'distance': distance,
            'zone': self.classify_zone(distance),
            'safety_level': self.determine_safety_level(distance),
            'recommended_action': self.get_recommended_action(distance)
        }

        return safety_assessment

    def classify_zone(self, distance):
        """
        Classify distance according to proxemics zones
        """
        if distance <= self.intimate_zone:
            return "intimate"
        elif distance <= self.personal_zone:
            return "personal"
        elif distance <= self.social_zone:
            return "social"
        else:
            return "public"

    def determine_safety_level(self, distance):
        """
        Determine safety level based on distance
        """
        if distance <= self.intimate_zone:
            return "critical"  # Too close, immediate action needed
        elif distance <= self.personal_zone:
            return "caution"   # Enter personal zone only if necessary
        elif distance <= self.social_zone:
            return "safe"      # Normal interaction distance
        else:
            return "distant"   # Adequate safety distance

    def get_recommended_action(self, distance):
        """
        Get recommended action based on distance
        """
        if distance <= self.intimate_zone:
            return "IMMEDIATE RETREAT"
        elif distance <= self.personal_zone:
            return "MAINTAIN CAUTION"
        elif distance <= self.social_zone:
            return "NORMAL INTERACTION"
        else:
            return "APPROACH IF NEEDED"
```

## Functional Safety Standards

### IEC 61508 and ISO 13482 for Service Robots

International standards for robot safety:

#### Safety Integrity Levels (SIL)
```python
class SafetyIntegrityChecker:
    def __init__(self):
        self.sil_levels = {
            'SIL_1': {'pfhd': 1e-5, 'requirements': ['basic']},
            'SIL_2': {'pfhd': 1e-6, 'requirements': ['basic', 'redundancy']},
            'SIL_3': {'pfhd': 1e-7, 'requirements': ['basic', 'redundancy', 'diversity']},
            'SIL_4': {'pfhd': 1e-8, 'requirements': ['basic', 'redundancy', 'diversity', 'independence']}
        }

    def calculate_sil_requirement(self, robot_application):
        """
        Calculate required SIL level based on application
        """
        risk_assessment = self.assess_application_risk(robot_application)

        # Determine SIL based on risk
        if risk_assessment['severity'] >= 4 and risk_assessment['exposure'] >= 3:
            return 'SIL_4'
        elif risk_assessment['severity'] >= 3 and risk_assessment['exposure'] >= 2:
            return 'SIL_3'
        elif risk_assessment['severity'] >= 2 and risk_assessment['exposure'] >= 2:
            return 'SIL_2'
        else:
            return 'SIL_1'

    def assess_application_risk(self, application):
        """
        Assess risk based on severity and exposure factors
        """
        return {
            'severity': application.get('severity', 1),  # 1-4 scale
            'exposure': application.get('exposure', 1),  # 1-4 scale
            'probability': application.get('probability', 1)  # 1-4 scale
        }

    def verify_safety_requirements(self, current_sil, target_sil):
        """
        Verify that current safety measures meet target SIL
        """
        if self.sil_levels[current_sil]['pfhd'] <= self.sil_levels[target_sil]['pfhd']:
            return True

        required_features = self.sil_levels[target_sil]['requirements']
        current_features = self.get_current_safety_features()

        missing_features = set(required_features) - set(current_features)
        return len(missing_features) == 0
```

### ISO 10218 for Robot Safety

Standards specifically for industrial robots that apply to humanoid systems:

#### Safety-Rated Monitoring
```python
class SafetyRatedMonitor:
    def __init__(self):
        self.monitored_variables = [
            'joint_positions',
            'joint_velocities',
            'joint_torques',
            'cartesian_positions',
            'external_forces',
            'environment_sensors'
        ]
        self.safety_limits = self.define_safety_limits()

    def define_safety_limits(self):
        """
        Define safety limits for monitored variables
        """
        return {
            'joint_positions': {
                'upper': np.array([1.57, 1.57, 1.57, 1.57, 1.57, 1.57]),  # rad
                'lower': np.array([-1.57, -1.57, -1.57, -1.57, -1.57, -1.57])  # rad
            },
            'joint_velocities': {
                'max': np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # rad/s
            },
            'joint_torques': {
                'max': np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # Nm
            },
            'cartesian_positions': {
                'workspace_limits': {
                    'x': (-2.0, 2.0),
                    'y': (-2.0, 2.0),
                    'z': (0.0, 2.0)
                }
            }
        }

    def monitor_safety_variables(self, robot_state):
        """
        Monitor safety-rated variables in real-time
        """
        violations = []

        # Check joint position limits
        joint_pos_violations = self.check_joint_position_limits(
            robot_state['joint_positions']
        )
        violations.extend(joint_pos_violations)

        # Check joint velocity limits
        joint_vel_violations = self.check_joint_velocity_limits(
            robot_state['joint_velocities']
        )
        violations.extend(joint_vel_violations)

        # Check torque limits
        torque_violations = self.check_torque_limits(
            robot_state['joint_torques']
        )
        violations.extend(torque_violations)

        # Check workspace limits
        workspace_violations = self.check_workspace_limits(
            robot_state['cartesian_position']
        )
        violations.extend(workspace_violations)

        return violations

    def check_joint_position_limits(self, joint_positions):
        """
        Check if joint positions are within limits
        """
        violations = []
        limits = self.safety_limits['joint_positions']

        for i, pos in enumerate(joint_positions):
            if pos > limits['upper'][i] or pos < limits['lower'][i]:
                violations.append({
                    'type': 'joint_position',
                    'joint': i,
                    'value': pos,
                    'limit': f"{limits['lower'][i]} to {limits['upper'][i]}",
                    'severity': 'high'
                })

        return violations
```

## Ethical Considerations

### Robot Ethics Frameworks

Ethical frameworks guide the development and deployment of humanoid robots:

#### Asimov's Laws Modern Interpretation
```python
class RobotEthicsFramework:
    def __init__(self):
        # Modern interpretation of robot ethics principles
        self.principles = [
            "Beneficence: Act in ways that promote human welfare",
            "Non-maleficence: Do no harm to humans",
            "Autonomy: Respect human decision-making authority",
            "Justice: Treat humans fairly and equitably",
            "Explicability: Be transparent about capabilities and limitations"
        ]

    def evaluate_action_ethics(self, proposed_action, context):
        """
        Evaluate if an action is ethically acceptable
        """
        ethical_assessment = {
            'action': proposed_action,
            'beneficence_score': self.assess_beneficence(proposed_action, context),
            'non_maleficence_score': self.assess_non_maleficence(proposed_action, context),
            'autonomy_score': self.assess_autonomy_respect(proposed_action, context),
            'justice_score': self.assess_justice(proposed_action, context),
            'explicability_score': self.assess_explicability(proposed_action, context),
            'overall_ethical_score': 0
        }

        # Calculate weighted overall score
        weights = {
            'beneficence': 0.25,
            'non_maleficence': 0.30,  # Highest weight
            'autonomy': 0.20,
            'justice': 0.15,
            'explicability': 0.10
        }

        overall_score = sum(
            ethical_assessment[f"{principle}_score"] * weights[principle]
            for principle in weights
        )

        ethical_assessment['overall_ethical_score'] = overall_score
        ethical_assessment['acceptable'] = overall_score >= 0.7  # 70% threshold

        return ethical_assessment

    def assess_non_maleficence(self, action, context):
        """
        Assess potential harm from the action
        """
        # Analyze potential negative consequences
        harm_risk = 0.0

        # Physical harm risk
        if self.would_cause_physical_harm(action, context):
            harm_risk += 0.8

        # Psychological harm risk
        if self.would_cause_psychological_harm(action, context):
            harm_risk += 0.6

        # Privacy violation risk
        if self.would_violate_privacy(action, context):
            harm_risk += 0.4

        # Return score (0 = no harm, 1 = maximum harm)
        return 1.0 - min(harm_risk, 1.0)

    def would_cause_physical_harm(self, action, context):
        """
        Check if action would cause physical harm
        """
        # Analyze action for potential physical harm
        return False

    def would_cause_psychological_harm(self, action, context):
        """
        Check if action would cause psychological harm
        """
        # Analyze action for potential psychological impact
        return False

    def would_violate_privacy(self, action, context):
        """
        Check if action would violate privacy
        """
        # Analyze action for potential privacy violations
        return False
```

### Privacy and Data Protection

Humanoid robots collect sensitive data requiring protection:

#### Privacy-Preserving Architecture
```python
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from datetime import datetime, timedelta

class PrivacyManager:
    def __init__(self):
        self.encryption_key = self.generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.consent_records = {}
        self.data_retention_policies = self.define_retention_policies()

    def generate_encryption_key(self):
        """
        Generate secure encryption key
        """
        return Fernet.generate_key()

    def encrypt_sensitive_data(self, data):
        """
        Encrypt sensitive data before storage
        """
        if isinstance(data, str):
            data = data.encode()
        return self.cipher_suite.encrypt(data)

    def anonymize_data(self, personal_data):
        """
        Anonymize personal data while preserving utility
        """
        anonymized = {}

        for key, value in personal_data.items():
            if key in ['name', 'id', 'email']:
                # Replace with pseudonym
                anonymized[key] = self.create_pseudonym(value)
            elif key in ['location', 'face_image']:
                # Apply differential privacy
                anonymized[key] = self.apply_differential_privacy(value)
            else:
                # Keep other data as is
                anonymized[key] = value

        return anonymized

    def create_pseudonym(self, original_value):
        """
        Create pseudonym for personal identifiers
        """
        salt = secrets.token_bytes(16)
        pseudonym = hashlib.sha256(salt + original_value.encode()).hexdigest()
        return pseudonym

    def apply_differential_privacy(self, data, epsilon=1.0):
        """
        Apply differential privacy to data
        """
        # Add noise to preserve privacy
        if isinstance(data, (int, float)):
            noise = np.random.laplace(0, 1/epsilon)
            return data + noise
        return data

    def manage_consent(self, user_id, data_types, consent_granted=True):
        """
        Manage user consent for data collection
        """
        consent_record = {
            'timestamp': datetime.now(),
            'data_types': data_types,
            'granted': consent_granted,
            'revocable': True
        }

        if user_id not in self.consent_records:
            self.consent_records[user_id] = []

        self.consent_records[user_id].append(consent_record)

    def enforce_data_retention(self):
        """
        Enforce data retention policies
        """
        current_time = datetime.now()

        for user_id, records in self.consent_records.items():
            for record in records:
                if current_time - record['timestamp'] > self.data_retention_policies['max_age']:
                    self.delete_user_data(user_id)

    def define_retention_policies(self):
        """
        Define data retention policies
        """
        return {
            'max_age': timedelta(days=365),  # Maximum retention period
            'sensitive_data_max_age': timedelta(days=30),  # Shorter for sensitive data
            'automated_purge': True,  # Automatically delete old data
            'user_deletion': True  # Honor user deletion requests
        }
```

## Fail-Safe Mechanisms

### Emergency Procedures

Humanoid robots must have reliable emergency procedures:

#### Emergency Stop System
```python
import signal
import threading
import time

class EmergencyStopSystem:
    def __init__(self):
        self.emergency_stop_triggered = False
        self.emergency_stop_reason = None
        self.safety_functions = []
        self.lock = threading.Lock()

    def register_safety_function(self, func, name):
        """
        Register a safety function to be called during emergency stop
        """
        self.safety_functions.append({'function': func, 'name': name})

    def trigger_emergency_stop(self, reason="Unknown"):
        """
        Trigger emergency stop with reason
        """
        with self.lock:
            if not self.emergency_stop_triggered:
                self.emergency_stop_triggered = True
                self.emergency_stop_reason = reason

                # Execute all registered safety functions
                for safety_func in self.safety_functions:
                    try:
                        safety_func['function']()
                    except Exception as e:
                        print(f"Error in safety function {safety_func['name']}: {e}")

                # Log the emergency stop
                self.log_emergency_stop(reason)

    def reset_emergency_stop(self):
        """
        Reset emergency stop state (after safety check)
        """
        with self.lock:
            if self.emergency_stop_triggered:
                # Verify it's safe to reset
                if self.verify_safe_to_resume():
                    self.emergency_stop_triggered = False
                    self.emergency_stop_reason = None
                    return True
                else:
                    return False

    def verify_safe_to_resume(self):
        """
        Verify it's safe to resume operation after emergency stop
        """
        # Check that all safety-critical systems are nominal
        # Check that environment is safe
        # Check that robot is in safe configuration
        return True

    def log_emergency_stop(self, reason):
        """
        Log emergency stop event for analysis
        """
        log_entry = {
            'timestamp': time.time(),
            'reason': reason,
            'robot_state': self.get_robot_state(),
            'environment_state': self.get_environment_state(),
            'safety_functions_executed': [f['name'] for f in self.safety_functions]
        }

        # Write to safety log
        self.write_safety_log(log_entry)

    def get_robot_state(self):
        """
        Get current robot state for logging
        """
        return {
            'joint_positions': [],
            'joint_velocities': [],
            'cartesian_pose': None,
            'operational_mode': 'normal'
        }
```

### Graceful Degradation

Systems should fail safely rather than catastrophically:

#### Degradation Manager
```python
class DegradationManager:
    def __init__(self):
        self.system_levels = [
            'full_functionality',
            'reduced_functionality',
            'safe_mode',
            'emergency_stop'
        ]
        self.current_level = 'full_functionality'
        self.degradation_thresholds = self.define_degradation_thresholds()

    def define_degradation_thresholds(self):
        """
        Define thresholds for system degradation
        """
        return {
            'full_functionality': {
                'battery': (0.2, 1.0),  # 20-100% battery
                'temperature': (10, 60),  # 10-60°C
                'network': True,  # Network available
                'sensors': 0.95  # 95%+ sensors functional
            },
            'reduced_functionality': {
                'battery': (0.1, 0.2),   # 10-20% battery
                'temperature': (5, 70),  # 5-70°C
                'network': False,  # Network optional
                'sensors': 0.80  # 80%+ sensors functional
            },
            'safe_mode': {
                'battery': (0.05, 0.1),  # 5-10% battery
                'temperature': (0, 80),  # 0-80°C
                'network': False,
                'sensors': 0.50  # 50%+ sensors functional
            },
            'emergency_stop': {
                'battery': (0.0, 0.05),  # <5% battery
                'temperature': (80, float('inf')),  # >80°C
                'sensors': 0.0  # No sensors functional
            }
        }

    def assess_system_state(self, system_status):
        """
        Assess current system state and determine appropriate level
        """
        current_level = 'full_functionality'

        for level in reversed(self.system_levels):
            if self.meets_level_requirements(level, system_status):
                current_level = level
                break

        return current_level

    def meets_level_requirements(self, level, system_status):
        """
        Check if system meets requirements for a degradation level
        """
        requirements = self.degradation_thresholds[level]

        # Check battery level
        if 'battery_level' in system_status:
            battery_ok = (requirements['battery'][0] <=
                         system_status['battery_level'] <=
                         requirements['battery'][1])
            if not battery_ok:
                return False

        # Check temperature
        if 'temperature' in system_status:
            temp_ok = (requirements['temperature'][0] <=
                      system_status['temperature'] <=
                      requirements['temperature'][1])
            if not temp_ok:
                return False

        # Check network availability
        if 'network_available' in system_status:
            network_ok = (system_status['network_available'] == requirements['network'])
            if not network_ok:
                return False

        # Check sensor functionality
        if 'functional_sensors_ratio' in system_status:
            sensor_ok = system_status['functional_sensors_ratio'] >= requirements['sensors']
            if not sensor_ok:
                return False

        return True

    def apply_degradation(self, new_level):
        """
        Apply system degradation to new level
        """
        if self.system_levels.index(new_level) < self.system_levels.index(self.current_level):
            # Moving to lower level - apply restrictions
            self.current_level = new_level
            self.implement_level_restrictions(new_level)

    def implement_level_restrictions(self, level):
        """
        Implement restrictions for a degradation level
        """
        restrictions = {
            'reduced_functionality': [
                'limit_speed',
                'reduce_payload',
                'disable_complex_behaviors'
            ],
            'safe_mode': [
                'minimal_movement',
                'return_to_home',
                'wait_for_assistance'
            ],
            'emergency_stop': [
                'complete_stop',
                'preserve_critical_systems',
                'await_manual_reset'
            ]
        }

        if level in restrictions:
            for restriction in restrictions[level]:
                self.apply_restriction(restriction)

    def apply_restriction(self, restriction):
        """
        Apply specific restriction
        """
        # Implement the specific restriction
        pass
```

## Security Considerations

### Cybersecurity for Humanoid Robots

Humanoid robots connected to networks require robust cybersecurity:

#### Authentication and Authorization
```python
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta

class RobotSecurityManager:
    def __init__(self):
        self.secret_key = secrets.token_urlsafe(32)
        self.access_control_list = {}
        self.session_tokens = {}

    def authenticate_user(self, username, password, device_id):
        """
        Authenticate user with device verification
        """
        # Verify credentials
        if not self.verify_credentials(username, password):
            return None

        # Verify device
        if not self.verify_device(device_id, username):
            return None

        # Generate access token
        token = self.generate_access_token(username, device_id)
        return token

    def verify_credentials(self, username, password):
        """
        Verify user credentials against stored hash
        """
        # Retrieve stored hash and salt
        stored_hash, salt = self.get_stored_credentials(username)

        # Hash provided password with stored salt
        provided_hash = hashlib.pbkdf2_hmac(
            'sha256', password.encode(), salt, 100000
        )

        return hmac.compare_digest(stored_hash, provided_hash)

    def generate_access_token(self, username, device_id):
        """
        Generate JWT access token with expiration
        """
        payload = {
            'username': username,
            'device_id': device_id,
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }

        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return token

    def authorize_action(self, token, action, resource):
        """
        Authorize specific action on resource
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            username = payload['username']

            # Check permissions
            if self.check_permission(username, action, resource):
                return True
            else:
                self.log_unauthorized_access(username, action, resource)
                return False

        except jwt.ExpiredSignatureError:
            self.log_expired_token(username, action, resource)
            return False
        except jwt.InvalidTokenError:
            self.log_invalid_token(username, action, resource)
            return False

    def check_permission(self, username, action, resource):
        """
        Check if user has permission for action on resource
        """
        # Check ACL for user permissions
        user_permissions = self.access_control_list.get(username, {})
        resource_permissions = user_permissions.get(resource, [])

        return action in resource_permissions
```

### Secure Communication

#### Encrypted Communication Protocol
```python
import ssl
import socket
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

class SecureCommunication:
    def __init__(self, certificate_path, private_key_path):
        self.certificate_path = certificate_path
        self.private_key_path = private_key_path
        self.session_keys = {}

    def establish_secure_connection(self, host, port):
        """
        Establish TLS-encrypted connection
        """
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.check_hostname = False  # For robot-to-robot communication
        context.load_cert_chain(self.certificate_path, self.private_key_path)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        secure_sock = context.wrap_socket(sock, server_hostname=host)

        secure_sock.connect((host, port))
        return secure_sock

    def encrypt_message(self, message, recipient_public_key):
        """
        Encrypt message for secure transmission
        """
        # Generate random symmetric key for this message
        symmetric_key = secrets.token_bytes(32)  # AES-256 key

        # Encrypt message with symmetric key
        iv = secrets.token_bytes(16)  # AES block size
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
        encryptor = cipher.encryptor()

        # Pad message to block size
        padded_message = self.pad_message(message.encode())
        encrypted_message = encryptor.update(padded_message) + encryptor.finalize()

        # Encrypt symmetric key with recipient's public key
        encrypted_key = recipient_public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return {
            'encrypted_message': encrypted_message,
            'encrypted_key': encrypted_key,
            'iv': iv
        }

    def pad_message(self, message):
        """
        Pad message to AES block size (16 bytes)
        """
        block_size = 16
        padding_length = block_size - (len(message) % block_size)
        padding = bytes([padding_length] * padding_length)
        return message + padding
```

## Safety Validation and Testing

### Safety Testing Framework

Comprehensive testing ensures safety systems work correctly:

#### Safety Test Suite
```python
import unittest
import numpy as np

class SafetyTestSuite(unittest.TestCase):
    def setUp(self):
        self.safety_controller = SafetyController()
        self.collision_detector = CollisionDetector()
        self.emergency_stop = EmergencyStopSystem()

    def test_force_limiting(self):
        """
        Test that force limits are properly enforced
        """
        # Test normal operation
        normal_torques = [10.0, 15.0, 20.0]  # Within limits
        normal_forces = [np.array([10.0, 5.0, 2.0])]  # Within limits

        result = self.safety_controller.check_force_limits(normal_torques, normal_forces)
        self.assertTrue(result, "Normal forces should be allowed")

        # Test force violation
        high_torques = [150.0, 200.0, 120.0]  # Exceeding limits
        high_forces = [np.array([100.0, 80.0, 60.0])]  # Exceeding limits

        result = self.safety_controller.check_force_limits(high_torques, high_forces)
        self.assertFalse(result, "High forces should trigger safety stop")

    def test_collision_detection(self):
        """
        Test collision detection functionality
        """
        # Create test scenario
        robot_pose = np.array([0, 0, 1, 0, 0, 0, 1])  # [x, y, z, qx, qy, qz, qw]
        env_points = np.array([[0.5, 0, 1], [0, 0.5, 1]])  # Points near robot

        collision_risk = self.collision_detector.detect_collisions(
            robot_pose, env_points, humans=None
        )

        self.assertTrue(collision_risk['environment'], "Should detect nearby obstacles")

    def test_emergency_stop(self):
        """
        Test emergency stop functionality
        """
        # Initially not triggered
        self.assertFalse(self.emergency_stop.emergency_stop_triggered)

        # Trigger emergency stop
        self.emergency_stop.trigger_emergency_stop("Test reason")

        self.assertTrue(self.emergency_stop.emergency_stop_triggered)

        # Reset emergency stop (assuming safe conditions)
        reset_success = self.emergency_stop.reset_emergency_stop()
        self.assertTrue(reset_success)

    def test_safety_integrity(self):
        """
        Test safety integrity verification
        """
        checker = SafetyIntegrityChecker()

        # Test SIL verification
        application = {
            'severity': 3,
            'exposure': 3,
            'probability': 2
        }

        required_sil = checker.calculate_sil_requirement(application)
        self.assertIn(required_sil, ['SIL_3', 'SIL_4'])

if __name__ == '__main__':
    unittest.main()
```

## Regulatory Compliance

### Standards and Certification

Humanoid robots must comply with various regulatory standards:

#### Compliance Tracking System
```python
class ComplianceTracker:
    def __init__(self):
        self.standards = {
            'ISO 13482': {'status': 'pending', 'requirements': [], 'tests_passed': 0},
            'ISO 10218': {'status': 'pending', 'requirements': [], 'tests_passed': 0},
            'IEC 61508': {'status': 'pending', 'requirements': [], 'tests_passed': 0},
            'GDPR': {'status': 'pending', 'requirements': [], 'tests_passed': 0}
        }
        self.certification_history = []

    def assess_compliance(self, standard_name):
        """
        Assess compliance with specific standard
        """
        if standard_name not in self.standards:
            raise ValueError(f"Unknown standard: {standard_name}")

        # Perform compliance assessment
        requirements_met = self.check_requirements(standard_name)
        tests_passed = self.count_passed_tests(standard_name)

        compliance_score = self.calculate_compliance_score(
            requirements_met, tests_passed, standard_name
        )

        self.standards[standard_name].update({
            'compliance_score': compliance_score,
            'requirements_met': requirements_met,
            'tests_passed': tests_passed,
            'status': 'compliant' if compliance_score >= 0.95 else 'non_compliant'
        })

        return self.standards[standard_name]

    def check_requirements(self, standard_name):
        """
        Check if requirements for standard are met
        """
        # Implementation would check specific requirements
        # for the given standard
        return 0.0

    def calculate_compliance_score(self, requirements_met, tests_passed, standard_name):
        """
        Calculate overall compliance score
        """
        # Weighted calculation based on standard requirements
        return min(requirements_met, tests_passed / 100.0)  # Simplified calculation
```

## Summary

Safety and ethics in humanoid robotics encompass physical safety, cybersecurity, privacy protection, and ethical decision-making. Through robust safety systems, comprehensive testing, and adherence to international standards, humanoid robots can operate safely in human environments. The integration of safety considerations from the design phase through deployment ensures that these advanced systems enhance human life while minimizing risks. As humanoid robotics continues to advance, maintaining the highest safety and ethical standards will be essential for public acceptance and successful deployment.