"""Unit tests for shp.
"""

import os
import tempfile
import pytest

ogr = pytest.importorskip("osgeo.ogr")

import networkx as nx


class TestShp:
    def setup_method(self):
        def createlayer(driver, layerType=ogr.wkbLineString):
            lyr = driver.CreateLayer("edges", None, layerType)
            namedef = ogr.FieldDefn("Name", ogr.OFTString)
            namedef.SetWidth(32)
            lyr.CreateField(namedef)
            return lyr

        drv = ogr.GetDriverByName("ESRI Shapefile")

        testdir = os.path.join(tempfile.gettempdir(), "shpdir")
        shppath = os.path.join(tempfile.gettempdir(), "tmpshp.shp")
        multi_shppath = os.path.join(tempfile.gettempdir(), "tmp_mshp.shp")

        self.deletetmp(drv, testdir, shppath, multi_shppath)
        os.mkdir(testdir)

        self.names = ["a", "b", "c", "c"]  # edgenames
        self.paths = (
            [(1.0, 1.0), (2.0, 2.0)],
            [(2.0, 2.0), (3.0, 3.0)],
            [(0.9, 0.9), (4.0, 0.9), (4.0, 2.0)],
        )

        self.simplified_names = ["a", "b", "c"]  # edgenames
        self.simplified_paths = (
            [(1.0, 1.0), (2.0, 2.0)],
            [(2.0, 2.0), (3.0, 3.0)],
            [(0.9, 0.9), (4.0, 2.0)],
        )

        self.multi_names = ["a", "a", "a", "a"]  # edgenames

        shp = drv.CreateDataSource(shppath)
        lyr = createlayer(shp)

        for path, name in zip(self.paths, self.names):
            feat = ogr.Feature(lyr.GetLayerDefn())
            g = ogr.Geometry(ogr.wkbLineString)
            for p in path:
                g.AddPoint_2D(*p)
            feat.SetGeometry(g)
            feat.SetField("Name", name)
            lyr.CreateFeature(feat)

        # create single record multiline shapefile for testing
        multi_shp = drv.CreateDataSource(multi_shppath)
        multi_lyr = createlayer(multi_shp, ogr.wkbMultiLineString)

        multi_g = ogr.Geometry(ogr.wkbMultiLineString)
        for path in self.paths:

            g = ogr.Geometry(ogr.wkbLineString)
            for p in path:
                g.AddPoint_2D(*p)

            multi_g.AddGeometry(g)

        multi_feat = ogr.Feature(multi_lyr.GetLayerDefn())
        multi_feat.SetGeometry(multi_g)
        multi_feat.SetField("Name", "a")
        multi_lyr.CreateFeature(multi_feat)

        self.shppath = shppath
        self.multi_shppath = multi_shppath
        self.testdir = testdir
        self.drv = drv

    def deletetmp(self, drv, *paths):
        for p in paths:
            if os.path.exists(p):
                drv.DeleteDataSource(p)

    def testload(self):
        def compare_graph_paths_names(g, paths, names):
            expected = nx.DiGraph()
            for p in paths:
                nx.add_path(expected, p)
            assert sorted(expected.nodes) == sorted(g.nodes)
            assert sorted(expected.edges()) == sorted(g.edges())
            g_names = [g.get_edge_data(s, e)["Name"] for s, e in g.edges()]
            assert names == sorted(g_names)

        # simplified
        G = nx.read_shp(self.shppath)
        compare_graph_paths_names(G, self.simplified_paths, self.simplified_names)

        # unsimplified
        G = nx.read_shp(self.shppath, simplify=False)
        compare_graph_paths_names(G, self.paths, self.names)

        # multiline unsimplified
        G = nx.read_shp(self.multi_shppath, simplify=False)
        compare_graph_paths_names(G, self.paths, self.multi_names)

    def checkgeom(self, lyr, expected):
        feature = lyr.GetNextFeature()
        actualwkt = []
        while feature:
            actualwkt.append(feature.GetGeometryRef().ExportToWkt())
            feature = lyr.GetNextFeature()
        assert sorted(expected) == sorted(actualwkt)

    def test_geometryexport(self):
        expectedpoints_simple = (
            "POINT (1 1)",
            "POINT (2 2)",
            "POINT (3 3)",
            "POINT (0.9 0.9)",
            "POINT (4 2)",
        )
        expectedlines_simple = (
            "LINESTRING (1 1,2 2)",
            "LINESTRING (2 2,3 3)",
            "LINESTRING (0.9 0.9,4.0 0.9,4 2)",
        )
        expectedpoints = (
            "POINT (1 1)",
            "POINT (2 2)",
            "POINT (3 3)",
            "POINT (0.9 0.9)",
            "POINT (4.0 0.9)",
            "POINT (4 2)",
        )
        expectedlines = (
            "LINESTRING (1 1,2 2)",
            "LINESTRING (2 2,3 3)",
            "LINESTRING (0.9 0.9,4.0 0.9)",
            "LINESTRING (4.0 0.9,4 2)",
        )

        tpath = os.path.join(tempfile.gettempdir(), "shpdir")
        G = nx.read_shp(self.shppath)
        nx.write_shp(G, tpath)
        shpdir = ogr.Open(tpath)
        self.checkgeom(shpdir.GetLayerByName("nodes"), expectedpoints_simple)
        self.checkgeom(shpdir.GetLayerByName("edges"), expectedlines_simple)

        # Test unsimplified
        # Nodes should have additional point,
        # edges should be 'flattened'
        G = nx.read_shp(self.shppath, simplify=False)
        nx.write_shp(G, tpath)
        shpdir = ogr.Open(tpath)
        self.checkgeom(shpdir.GetLayerByName("nodes"), expectedpoints)
        self.checkgeom(shpdir.GetLayerByName("edges"), expectedlines)

    def test_attributeexport(self):
        def testattributes(lyr, graph):
            feature = lyr.GetNextFeature()
            while feature:
                coords = []
                ref = feature.GetGeometryRef()
                last = ref.GetPointCount() - 1
                edge_nodes = (ref.GetPoint_2D(0), ref.GetPoint_2D(last))
                name = feature.GetFieldAsString("Name")
                assert graph.get_edge_data(*edge_nodes)["Name"] == name
                feature = lyr.GetNextFeature()

        tpath = os.path.join(tempfile.gettempdir(), "shpdir")

        G = nx.read_shp(self.shppath)
        nx.write_shp(G, tpath)
        shpdir = ogr.Open(tpath)
        edges = shpdir.GetLayerByName("edges")
        testattributes(edges, G)

    # Test export of node attributes in nx.write_shp (#2778)
    def test_nodeattributeexport(self):
        tpath = os.path.join(tempfile.gettempdir(), "shpdir")

        G = nx.DiGraph()
        A = (0, 0)
        B = (1, 1)
        C = (2, 2)
        G.add_edge(A, B)
        G.add_edge(A, C)
        label = "node_label"
        for n, d in G.nodes(data=True):
            d["label"] = label
        nx.write_shp(G, tpath)

        H = nx.read_shp(tpath)
        for n, d in H.nodes(data=True):
            assert d["label"] == label

    def test_wkt_export(self):
        G = nx.DiGraph()
        tpath = os.path.join(tempfile.gettempdir(), "shpdir")
        points = ("POINT (0.9 0.9)", "POINT (4 2)")
        line = ("LINESTRING (0.9 0.9,4 2)",)
        G.add_node(1, Wkt=points[0])
        G.add_node(2, Wkt=points[1])
        G.add_edge(1, 2, Wkt=line[0])
        try:
            nx.write_shp(G, tpath)
        except Exception as e:
            assert False, e
        shpdir = ogr.Open(tpath)
        self.checkgeom(shpdir.GetLayerByName("nodes"), points)
        self.checkgeom(shpdir.GetLayerByName("edges"), line)

    def teardown_method(self):
        self.deletetmp(self.drv, self.testdir, self.shppath)


def test_read_shp_nofile():
    with pytest.raises(RuntimeError):
        G = nx.read_shp("hopefully_this_file_will_not_be_available")


class TestMissingGeometry:
    def setup_method(self):
        self.setup_path()
        self.delete_shapedir()
        self.create_shapedir()

    def teardown_method(self):
        self.delete_shapedir()

    def setup_path(self):
        self.path = os.path.join(tempfile.gettempdir(), "missing_geometry")

    def create_shapedir(self):
        drv = ogr.GetDriverByName("ESRI Shapefile")
        shp = drv.CreateDataSource(self.path)
        lyr = shp.CreateLayer("nodes", None, ogr.wkbPoint)
        feature = ogr.Feature(lyr.GetLayerDefn())
        feature.SetGeometry(None)
        lyr.CreateFeature(feature)
        feature.Destroy()

    def delete_shapedir(self):
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(self.path):
            drv.DeleteDataSource(self.path)

    def test_missing_geometry(self):
        with pytest.raises(nx.NetworkXError):
            G = nx.read_shp(self.path)


class TestMissingAttrWrite:
    def setup_method(self):
        self.setup_path()
        self.delete_shapedir()

    def teardown_method(self):
        self.delete_shapedir()

    def setup_path(self):
        self.path = os.path.join(tempfile.gettempdir(), "missing_attributes")

    def delete_shapedir(self):
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(self.path):
            drv.DeleteDataSource(self.path)

    def test_missing_attributes(self):
        G = nx.DiGraph()
        A = (0, 0)
        B = (1, 1)
        C = (2, 2)
        G.add_edge(A, B, foo=100)
        G.add_edge(A, C)

        nx.write_shp(G, self.path)
        H = nx.read_shp(self.path)

        for u, v, d in H.edges(data=True):
            if u == A and v == B:
                assert d["foo"] == 100
            if u == A and v == C:
                assert d["foo"] is None
