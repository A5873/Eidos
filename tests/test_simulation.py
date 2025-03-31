import unittest
import numpy as np
from sim9 import (
    Simulation, Being, Environment, 
    QuantumState, GeneticProfile, 
    ConsciousnessProfile, RealityManipulator
)

class TestSimulation(unittest.TestCase):
    def setUp(self):
        """Set up test simulation with minimal configuration"""
        self.config = {
            "num_beings": 5,
            "universe_size": 50.0,
            "max_steps": 10,
            "visualization_interval": 2,
            "save_interval": 5,
            "interaction_radius": 10.0,
            "seed": 42
        }
        self.simulation = Simulation(config=self.config)

    def test_simulation_initialization(self):
        """Test if simulation initializes correctly"""
        self.simulation.initialize()
        self.assertEqual(len(self.simulation.beings), self.config["num_beings"])
        self.assertIsInstance(self.simulation.environment, Environment)
        self.assertEqual(self.simulation.step_count, 0)

    def test_being_creation(self):
        """Test if beings are created with correct components"""
        being = Being()
        self.assertIsInstance(being.quantum_state, QuantumState)
        self.assertIsInstance(being.genetic_profile, GeneticProfile)
        self.assertIsInstance(being.consciousness_profile, ConsciousnessProfile)
        self.assertIsInstance(being.reality_manipulator, RealityManipulator)

    def test_quantum_state_evolution(self):
        """Test quantum state evolution"""
        quantum_state = QuantumState()
        initial_wave = quantum_state.wave_function.copy()
        
        # Create test Hamiltonian
        h_size = quantum_state.wave_function.shape[1]
        hamiltonian = np.eye(h_size) * 0.1
        
        # Evolve state
        quantum_state.evolve(hamiltonian, dt=0.1)
        
        # Check if state changed
        self.assertFalse(np.array_equal(quantum_state.wave_function, initial_wave))

    def test_simulation_step(self):
        """Test if simulation can execute steps"""
        self.simulation.initialize()
        initial_time = self.simulation.simulation_time
        
        # Execute one step
        step_results = self.simulation.step()
        
        # Check if time advanced
        self.assertGreater(self.simulation.simulation_time, initial_time)
        self.assertEqual(self.simulation.step_count, 1)
        self.assertIn("being_updates", step_results)

    def test_genetic_mutation(self):
        """Test genetic profile mutation"""
        genetic_profile = GeneticProfile(entity_id=None)
        initial_traits = genetic_profile.traits.copy()
        
        # Apply mutation
        mutations = genetic_profile.mutate()
        
        # Check if mutations occurred
        self.assertIsInstance(mutations, dict)
        if mutations:  # If mutations occurred
            self.assertNotEqual(genetic_profile.traits, initial_traits)

if __name__ == '__main__':
    unittest.main()
