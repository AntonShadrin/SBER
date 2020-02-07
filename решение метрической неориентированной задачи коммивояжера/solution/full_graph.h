#include <vector>
#include <map>
#include <stack>
#include <algorithm>

struct point
{
	double x;
	double y;
};

class FullGraph
{
	std::vector<point> vertexes_;
	std::multimap<double, std::pair<int, int>> distance_edges_;

	std::vector<int> parent_;
	std::vector<int> rank_;
	std::vector<std::vector<int>> SpanningTreeVertexes_;
public:
	struct v_ind_parent;

	explicit FullGraph(const std::vector<point>& vertexes);
	void CalculateDistance();

	double GetApproximateWeightGraphPath();
	double GetMinimalWeightGraphPath();

private:
	double GetDistance(const point& p1, const point& p2) const;
	int find(int element);
	bool merge(int left, int right);
};
