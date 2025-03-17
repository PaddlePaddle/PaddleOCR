/*******************************************************************************
 *                                                                              *
 * Author    :  Angus Johnson * Version   :  6.4.2 * Date      :  27 February
 *2017                                                * Website   :
 *http://www.angusj.com                                           * Copyright :
 *Angus Johnson 2010-2017                                         *
 *                                                                              *
 * License: * Use, modification & distribution is subject to Boost Software
 *License Ver 1. * http://www.boost.org/LICENSE_1_0.txt *
 *                                                                              *
 * Attributions: * The code in this library is an extension of Bala Vatti's
 *clipping algorithm: * "A generic solution to polygon clipping" *
 * Communications of the ACM, Vol 35, Issue 7 (July 1992) pp 56-63. *
 * http://portal.acm.org/citation.cfm?id=129906 *
 *                                                                              *
 * Computer graphics and geometric modeling: implementation and algorithms * By
 *Max K. Agoston                                                            *
 * Springer; 1 edition (January 4, 2005) *
 * http://books.google.com/books?q=vatti+clipping+agoston *
 *                                                                              *
 * See also: * "Polygon Offsetting by Computing Winding Numbers" * Paper no.
 *DETC2005-85513 pp. 565-575                                         * ASME 2005
 *International Design Engineering Technical Conferences             * and
 *Computers and Information in Engineering Conference (IDETC/CIE2005)      *
 * September 24-28, 2005 , Long Beach, California, USA *
 * http://www.me.berkeley.edu/~mcmains/pubs/DAC05OffsetPolygon.pdf *
 *                                                                              *
 *******************************************************************************/

#pragma once

#ifndef clipper_hpp
#define clipper_hpp

#define CLIPPER_VERSION "6.4.2"

// use_int32: When enabled 32bit ints are used instead of 64bit ints. This
// improve performance but coordinate values are limited to the range +/- 46340
//#define use_int32

// use_xyz: adds a Z member to IntPoint. Adds a minor cost to performance.
//#define use_xyz

// use_lines: Enables line clipping. Adds a very minor cost to performance.
#define use_lines

// use_deprecated: Enables temporary support for the obsolete functions
//#define use_deprecated

#include <list>
#include <queue>
#include <string>
#include <vector>

namespace ClipperLib {

enum ClipType { ctIntersection, ctUnion, ctDifference, ctXor };
enum PolyType { ptSubject, ptClip };
// By far the most widely used winding rules for polygon filling are
// EvenOdd & NonZero (GDI, GDI+, XLib, OpenGL, Cairo, AGG, Quartz, SVG, Gr32)
// Others rules include Positive, Negative and ABS_GTR_EQ_TWO (only in OpenGL)
// see http://glprogramming.com/red/chapter11.html
enum PolyFillType { pftEvenOdd, pftNonZero, pftPositive, pftNegative };

#ifdef use_int32
typedef int cInt;
static cInt const loRange = 0x7FFF;
static cInt const hiRange = 0x7FFF;
#else
typedef signed long long cInt;
static cInt const loRange = 0x3FFFFFFF;
static cInt const hiRange = 0x3FFFFFFFFFFFFFFFLL;
typedef signed long long long64; // used by Int128 class
typedef unsigned long long ulong64;

#endif

struct IntPoint {
  cInt X;
  cInt Y;
#ifdef use_xyz
  cInt Z;
  IntPoint(cInt x = 0, cInt y = 0, cInt z = 0) noexcept : X(x), Y(y), Z(z) {}
  IntPoint(IntPoint const &ip) noexcept : X(ip.X), Y(ip.Y), Z(ip.Z) {}
#else
  IntPoint(cInt x = 0, cInt y = 0) noexcept : X(x), Y(y) {}
  IntPoint(IntPoint const &ip) noexcept : X(ip.X), Y(ip.Y) {}
#endif

  inline void reset(cInt x = 0, cInt y = 0) noexcept {
    X = x;
    Y = y;
  }

  friend inline bool operator==(const IntPoint &a, const IntPoint &b) noexcept {
    return a.X == b.X && a.Y == b.Y;
  }
  friend inline bool operator!=(const IntPoint &a, const IntPoint &b) noexcept {
    return a.X != b.X || a.Y != b.Y;
  }
};
//------------------------------------------------------------------------------

typedef std::vector<IntPoint> Path;
typedef std::vector<Path> Paths;

inline Path &operator<<(Path &poly, IntPoint &&p) noexcept {
  poly.emplace_back(std::forward<IntPoint>(p));
  return poly;
}
inline Paths &operator<<(Paths &polys, Path &&p) noexcept {
  polys.emplace_back(std::forward<Path>(p));
  return polys;
}

std::ostream &operator<<(std::ostream &s, const IntPoint &p) noexcept;
std::ostream &operator<<(std::ostream &s, const Path &p) noexcept;
std::ostream &operator<<(std::ostream &s, const Paths &p) noexcept;

struct DoublePoint {
  double X;
  double Y;
  DoublePoint(double x = 0, double y = 0) noexcept : X(x), Y(y) {}
  DoublePoint(IntPoint const &ip) noexcept : X((double)ip.X), Y((double)ip.Y) {}
  inline void reset(double x = 0, double y = 0) noexcept {
    X = x;
    Y = y;
  }
};
//------------------------------------------------------------------------------

#ifdef use_xyz
typedef void (*ZFillCallback)(IntPoint &e1bot, IntPoint &e1top, IntPoint &e2bot,
                              IntPoint &e2top, IntPoint &pt);
#endif

enum InitOptions {
  ioReverseSolution = 1,
  ioStrictlySimple = 2,
  ioPreserveCollinear = 4
};
enum JoinType { jtSquare, jtRound, jtMiter };
enum EndType {
  etClosedPolygon,
  etClosedLine,
  etOpenButt,
  etOpenSquare,
  etOpenRound
};

class PolyNode;
typedef std::vector<PolyNode *> PolyNodes;

class PolyNode {
public:
  PolyNode() noexcept;
  virtual ~PolyNode() {}
  Path Contour;
  PolyNodes Children;
  PolyNode *Parent;
  PolyNode *GetNext() const noexcept;
  bool IsHole() const noexcept;
  bool IsOpen() const noexcept;
  int ChildCount() const noexcept;

private:
  // PolyNode& operator =(PolyNode& other);
  unsigned Index; // node index in Parent.Children
  bool m_IsOpen;
  JoinType m_jointype;
  EndType m_endtype;
  PolyNode *GetNextSiblingUp() const noexcept;
  void AddChild(PolyNode &child) noexcept;
  friend class Clipper; // to access Index
  friend class ClipperOffset;
};

class PolyTree : public PolyNode {
public:
  ~PolyTree() { Clear(); }
  PolyNode *GetFirst() const noexcept;
  void Clear() noexcept;
  int Total() const noexcept;

private:
  // PolyTree& operator =(PolyTree& other);
  PolyNodes AllNodes;
  friend class Clipper; // to access AllNodes
};

bool Orientation(const Path &poly) noexcept;
double Area(const Path &poly) noexcept;
int PointInPolygon(const IntPoint &pt, const Path &path) noexcept;

#if 0
void SimplifyPolygon(const Path &in_poly, Paths &out_polys,
                     PolyFillType fillType = pftEvenOdd);
void SimplifyPolygons(const Paths &in_polys, Paths &out_polys,
                      PolyFillType fillType = pftEvenOdd);
void SimplifyPolygons(Paths &polys, PolyFillType fillType = pftEvenOdd);
#endif

void CleanPolygon(const Path &in_poly, Path &out_poly,
                  double distance = 1.415) noexcept;
void CleanPolygon(Path &poly, double distance = 1.415) noexcept;
void CleanPolygons(const Paths &in_polys, Paths &out_polys,
                   double distance = 1.415) noexcept;
void CleanPolygons(Paths &polys, double distance = 1.415) noexcept;

#if 0
void MinkowskiSum(const Path &pattern, const Path &path, Paths &solution,
                  bool pathIsClosed);
void MinkowskiSum(const Path &pattern, const Paths &paths, Paths &solution,
                  bool pathIsClosed);

void MinkowskiDiff(const Path &poly1, const Path &poly2, Paths &solution);
#endif

void PolyTreeToPaths(const PolyTree &polytree, Paths &paths) noexcept;
void ClosedPathsFromPolyTree(const PolyTree &polytree, Paths &paths) noexcept;
void OpenPathsFromPolyTree(PolyTree &polytree, Paths &paths) noexcept;

void ReversePath(Path &p) noexcept;
void ReversePaths(Paths &p) noexcept;

struct IntRect {
  cInt left;
  cInt top;
  cInt right;
  cInt bottom;
};

// enums that are used internally ...
enum EdgeSide { esLeft = 1, esRight = 2 };

// forward declarations (for stuff used internally) ...
struct TEdge;
struct IntersectNode;
struct LocalMinimum;
struct OutPt;
struct OutRec;
struct Join;

typedef std::vector<OutRec *> PolyOutList;
typedef std::vector<TEdge *> EdgeList;
typedef std::vector<Join *> JoinList;
typedef std::vector<IntersectNode *> IntersectList;

//------------------------------------------------------------------------------

// ClipperBase is the ancestor to the Clipper class. It should not be
// instantiated directly. This class simply abstracts the conversion of sets of
// polygon coordinates into edge objects that are stored in a LocalMinima list.
class ClipperBase {
public:
  ClipperBase() noexcept;
  virtual ~ClipperBase();
  virtual bool AddPath(const Path &pg, PolyType PolyTyp, bool Closed);
  bool AddPaths(const Paths &ppg, PolyType PolyTyp, bool Closed);
  virtual void Clear() noexcept;
  IntRect GetBounds() noexcept;
  bool PreserveCollinear() const noexcept { return m_PreserveCollinear; }
  void PreserveCollinear(bool value) noexcept { m_PreserveCollinear = value; }

protected:
  void DisposeLocalMinimaList() noexcept;
  TEdge *AddBoundsToLML(TEdge *e, bool IsClosed) noexcept;
  virtual void Reset() noexcept;
  TEdge *ProcessBound(TEdge *E, bool IsClockwise) noexcept;
  void InsertScanbeam(const cInt Y) noexcept;
  bool PopScanbeam(cInt &Y) noexcept;
  bool LocalMinimaPending() noexcept;
  bool PopLocalMinima(cInt Y, const LocalMinimum *&locMin) noexcept;
  OutRec *CreateOutRec() noexcept;
  void DisposeAllOutRecs() noexcept;
  void DisposeOutRec(PolyOutList::size_type index) noexcept;
  void SwapPositionsInAEL(TEdge *edge1, TEdge *edge2) noexcept;
  void DeleteFromAEL(TEdge *e) noexcept;
  void UpdateEdgeIntoAEL(TEdge *&e);

  typedef std::vector<LocalMinimum> MinimaList;
  MinimaList::iterator m_CurrentLM;
  MinimaList m_MinimaList;

  bool m_UseFullRange;
  EdgeList m_edges;
  bool m_PreserveCollinear;
  bool m_HasOpenPaths;
  PolyOutList m_PolyOuts;
  TEdge *m_ActiveEdges;

  typedef std::priority_queue<cInt> ScanbeamList;
  ScanbeamList m_Scanbeam;
};
//------------------------------------------------------------------------------

class Clipper : public virtual ClipperBase {
public:
  Clipper(int initOptions = 0) noexcept;
  bool Execute(ClipType clipType, Paths &solution,
               PolyFillType fillType = pftEvenOdd);
  bool Execute(ClipType clipType, Paths &solution, PolyFillType subjFillType,
               PolyFillType clipFillType);
  bool Execute(ClipType clipType, PolyTree &polytree,
               PolyFillType fillType = pftEvenOdd) noexcept;
  bool Execute(ClipType clipType, PolyTree &polytree, PolyFillType subjFillType,
               PolyFillType clipFillType) noexcept;
  bool ReverseSolution() const noexcept { return m_ReverseOutput; }
  void ReverseSolution(bool value) noexcept { m_ReverseOutput = value; }
  bool StrictlySimple() const noexcept { return m_StrictSimple; }
  void StrictlySimple(bool value) noexcept { m_StrictSimple = value; }
// set the callback function for z value filling on intersections (otherwise Z
// is 0)
#ifdef use_xyz
  void ZFillFunction(ZFillCallback zFillFunc) noexcept;
#endif
protected:
  virtual bool ExecuteInternal() noexcept;

private:
  JoinList m_Joins;
  JoinList m_GhostJoins;
  IntersectList m_IntersectList;
  ClipType m_ClipType;
  typedef std::list<cInt> MaximaList;
  MaximaList m_Maxima;
  TEdge *m_SortedEdges;
  bool m_ExecuteLocked;
  PolyFillType m_ClipFillType;
  PolyFillType m_SubjFillType;
  bool m_ReverseOutput;
  bool m_UsingPolyTree;
  bool m_StrictSimple;
#ifdef use_xyz
  ZFillCallback m_ZFill; // custom callback
#endif
  void SetWindingCount(TEdge &edge) noexcept;
  bool IsEvenOddFillType(const TEdge &edge) const noexcept;
  bool IsEvenOddAltFillType(const TEdge &edge) const noexcept;
  void InsertLocalMinimaIntoAEL(const cInt botY) noexcept;
  void InsertEdgeIntoAEL(TEdge *edge, TEdge *startEdge) noexcept;
  void AddEdgeToSEL(TEdge *edge) noexcept;
  bool PopEdgeFromSEL(TEdge *&edge) noexcept;
  void CopyAELToSEL() noexcept;
  void DeleteFromSEL(TEdge *e) noexcept;
  void SwapPositionsInSEL(TEdge *edge1, TEdge *edge2) noexcept;
  bool IsContributing(const TEdge &edge) const noexcept;
  bool IsTopHorz(const cInt XPos) noexcept;
  void DoMaxima(TEdge *e);
  void ProcessHorizontals() noexcept;
  void ProcessHorizontal(TEdge *horzEdge) noexcept;
  void AddLocalMaxPoly(TEdge *e1, TEdge *e2, const IntPoint &pt) noexcept;
  OutPt *AddLocalMinPoly(TEdge *e1, TEdge *e2, const IntPoint &pt) noexcept;
  OutRec *GetOutRec(int idx) noexcept;
  void AppendPolygon(TEdge *e1, TEdge *e2) noexcept;
  void IntersectEdges(TEdge *e1, TEdge *e2, IntPoint &pt) noexcept;
  OutPt *AddOutPt(TEdge *e, const IntPoint &pt) noexcept;
  OutPt *GetLastOutPt(TEdge *e) noexcept;
  bool ProcessIntersections(const cInt topY);
  void BuildIntersectList(const cInt topY) noexcept;
  void ProcessIntersectList() noexcept;
  void ProcessEdgesAtTopOfScanbeam(const cInt topY);
  void BuildResult(Paths &polys) noexcept;
  void BuildResult2(PolyTree &polytree) noexcept;
  void SetHoleState(TEdge *e, OutRec *outrec) noexcept;
  void DisposeIntersectNodes() noexcept;
  bool FixupIntersectionOrder() noexcept;
  void FixupOutPolygon(OutRec &outrec) noexcept;
  void FixupOutPolyline(OutRec &outrec) noexcept;
  bool IsHole(TEdge *e) noexcept;
  bool FindOwnerFromSplitRecs(OutRec &outRec, OutRec *&currOrfl) noexcept;
  void FixHoleLinkage(OutRec &outrec) noexcept;
  void AddJoin(OutPt *op1, OutPt *op2, const IntPoint offPt) noexcept;
  void ClearJoins() noexcept;
  void ClearGhostJoins() noexcept;
  void AddGhostJoin(OutPt *op, const IntPoint offPt) noexcept;
  bool JoinPoints(Join *j, OutRec *outRec1, OutRec *outRec2) noexcept;
  void JoinCommonEdges() noexcept;
  void DoSimplePolygons() noexcept;
  void FixupFirstLefts1(OutRec *OldOutRec, OutRec *NewOutRec) noexcept;
  void FixupFirstLefts2(OutRec *InnerOutRec, OutRec *OuterOutRec) noexcept;
  void FixupFirstLefts3(OutRec *OldOutRec, OutRec *NewOutRec) noexcept;
#ifdef use_xyz
  void SetZ(IntPoint &pt, TEdge &e1, TEdge &e2) noexcept;
#endif
};
//------------------------------------------------------------------------------

class ClipperOffset {
public:
  ClipperOffset(double miterLimit = 2.0, double roundPrecision = 0.25) noexcept;
  ~ClipperOffset();
  void AddPath(const Path &path, JoinType joinType, EndType endType) noexcept;
  void AddPaths(const Paths &paths, JoinType joinType,
                EndType endType) noexcept;
  bool Execute(Paths &solution, double delta) noexcept;
  bool Execute(PolyTree &solution, double delta) noexcept;
  void Clear() noexcept;

private:
  double MiterLimit;
  double ArcTolerance;

  Paths m_destPolys;
  Path m_srcPoly;
  Path m_destPoly;
  std::vector<DoublePoint> m_normals;
  double m_delta, m_sinA, m_sin, m_cos;
  double m_miterLim, m_StepsPerRad;
  IntPoint m_lowest;
  PolyNode m_polyNodes;

  void FixOrientations() noexcept;
  void DoOffset(double delta) noexcept;
  void OffsetPoint(int j, int &k, JoinType jointype) noexcept;
  void DoSquare(int j, int k) noexcept;
  void DoMiter(int j, int k, double r) noexcept;
  void DoRound(int j, int k) noexcept;
};
//------------------------------------------------------------------------------

class clipperException : public std::exception {
public:
  clipperException(const char *description) noexcept : m_descr(description) {}
  ~clipperException() {}
  virtual const char *what() const noexcept { return m_descr.c_str(); }

private:
  std::string m_descr;
};
//------------------------------------------------------------------------------

} // namespace ClipperLib

#endif // clipper_hpp
