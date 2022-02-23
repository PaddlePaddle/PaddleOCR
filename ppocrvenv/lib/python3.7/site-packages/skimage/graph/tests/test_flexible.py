import numpy as np
import skimage.graph.mcp as mcp

from skimage._shared.testing import assert_array_equal


a = np.ones((8, 8), dtype=np.float32)
a[1::2] *= 2.0


class FlexibleMCP(mcp.MCP_Flexible):
    """ Simple MCP subclass that allows the front to travel 
    a certain distance from the seed point, and uses a constant
    cost factor that is independent of the cost array.
    """
    
    def _reset(self):
        mcp.MCP_Flexible._reset(self)
        self._distance = np.zeros((8, 8), dtype=np.float32).ravel()
    
    def goal_reached(self, index, cumcost):
        if self._distance[index] > 4:
            return 2
        else:
            return 0
    
    def travel_cost(self, index, new_index, offset_length):
        return 1.0  # fixed cost
    
    def examine_neighbor(self, index, new_index, offset_length):
        pass  # We do not test this
        
    def update_node(self, index, new_index, offset_length):
        self._distance[new_index] = self._distance[index] + 1


def test_flexible():
    # Create MCP and do a traceback
    mcp = FlexibleMCP(a)
    costs, traceback = mcp.find_costs([(0, 0)])
    
    # Check that inner part is correct. This basically
    # tests whether travel_cost works.
    assert_array_equal(costs[:4, :4], [[1, 2, 3, 4],
                                       [2, 2, 3, 4],
                                       [3, 3, 3, 4],
                                       [4, 4, 4, 4]])
    
    # Test that the algorithm stopped at the right distance.
    # Note that some of the costs are filled in but not yet frozen,
    # so we take a bit of margin
    assert np.all(costs[-2:, :] == np.inf)
    assert np.all(costs[:, -2:] == np.inf)
