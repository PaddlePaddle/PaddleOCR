import unittest
from shapely.geometry import Polygon
from tools.end2end.eval_end2end import calculate_iou, calculate_edit_distance, calculate_metrics, match_gt_and_dt

class TestEvalEnd2End(unittest.TestCase):

    def test_calculate_iou(self):
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        poly3 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        self.assertEqual(calculate_iou(poly1, poly2), 0)
        self.assertEqual(calculate_iou(poly1, poly3), 0.25)
        self.assertEqual(calculate_iou(poly1, poly1), 1)

    def test_calculate_edit_distance(self):
        self.assertEqual(calculate_edit_distance('test', 'test'), 0)
        self.assertEqual(calculate_edit_distance('test', 'tent'), 1)
        self.assertEqual(calculate_edit_distance('test', 'best'), 1)
        self.assertEqual(calculate_edit_distance('test', 'tests'), 1)
        self.assertEqual(calculate_edit_distance('test', 'testing'), 3)

    def test_calculate_metrics(self):
        self.assertEqual(calculate_metrics(5, 10, 10), (0.5, 0.5, 0.5))
        self.assertEqual(calculate_metrics(7, 10, 10), (0.7, 0.7, 0.7))
        self.assertEqual(calculate_metrics(10, 10, 10), (1, 1, 1))

    def test_match_gt_and_dt(self):
        # TODO: Implement this test based on the function's behavior and input/output

if __name__ == '__main__':
    unittest.main()
