# Aether Contract Layer — the "DNA" of valid blood cells.
#
# These Pydantic models define the strict JSON schema for all data crossing
# the C#-Python boundary. If C# sends malformed input, Pydantic catches it
# here before it can poison the ML pipeline. If Python returns malformed
# output, C# catches it via its own record-based validation (BloodFilter).
#
# Usage:
#   from contracts import CortexPredictInput, CortexPredictOutput, AetherResponse
