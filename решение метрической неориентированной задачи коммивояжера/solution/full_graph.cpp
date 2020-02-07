#include "full_graph.h"

struct FullGraph::v_ind_parent {
	v_ind_parent(int v_, int indx_, int parent_)
		: v(v_), indx(indx_), parent(parent_) {}
	int v;
	int indx;
	int parent;
};

FullGraph::FullGraph(const std::vector<point>& vertexes) : vertexes_(vertexes)
{
	SpanningTreeVertexes_.resize(vertexes.size());
}

void FullGraph::CalculateDistance()
{
	for (size_t i = 0; i < vertexes_.size() - 1; ++i)
	{
		for (size_t j = i + 1; j < vertexes_.size(); j++)
		{
			double distance = GetDistance(vertexes_[i], vertexes_[j]);
			std::pair<int, int> edge(i, j);
			distance_edges_.insert(std::pair<double, std::pair<int, int>>(distance, edge));
		}
	}
}

double FullGraph::GetDistance(const point& p1, const point& p2) const
{
	return pow(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2), 0.5);
}

double FullGraph::GetApproximateWeightGraphPath()
{
	parent_.resize(vertexes_.size(), -1);
	rank_.resize(vertexes_.size(), 1);
	std::stack<v_ind_parent> path;
	double len_path = 0.0;

	for (auto w_e : distance_edges_)
	{
		if (merge(w_e.second.first, w_e.second.second))
		{
			SpanningTreeVertexes_[w_e.second.first].push_back(w_e.second.second);
			SpanningTreeVertexes_[w_e.second.second].push_back(w_e.second.first);
		}
	}
	if (SpanningTreeVertexes_.size() > 0)
	{
		path.push(v_ind_parent(0, 0, -1));
		int prev_v = 0;
		while (!path.empty())
		{
			int v = path.top().v;
			int index = path.top().indx;
			int parent = path.top().parent;
			if (index < SpanningTreeVertexes_[v].size())
			{
				path.top().indx++;
				int child = SpanningTreeVertexes_[v][index];
				// skip parent
				if (child == parent)
					continue;

				len_path += GetDistance(vertexes_[prev_v], vertexes_[child]);
				prev_v = child;
				path.push(v_ind_parent(child, 0, v));
			}
			else
			{
				path.pop();
			}
		}
		len_path += GetDistance(vertexes_[prev_v], vertexes_[0]);
	}

	return len_path;
}

int FullGraph::find(int element) {
	if (parent_[element] == -1) {
		return element;
	}
	std::stack<int> q;
	while (parent_[element] != -1)
	{
		q.push(element);
		element = parent_[element];
	}
	while (!q.empty())
	{
		parent_[q.top()] = element;
		q.pop();
	}
	return element;
	//return parent[element] = find(parent[element]);
}

bool FullGraph::merge(int left, int right) {
	const int deputat_left = find(left);
	const int deputat_right = find(right);
	if (deputat_left == deputat_right)
	{
		return false;
	}
	if (rank_[deputat_left] == rank_[deputat_right]) {
		parent_[deputat_right] = deputat_left;
		++rank_[deputat_left];
	}
	else if (rank_[deputat_left] > rank_[deputat_right]) {
		parent_[deputat_right] = deputat_left;
	}
	else {
		parent_[deputat_left] = deputat_right;
	}
	return true;
}

double FullGraph::GetMinimalWeightGraphPath()
{
	std::vector<int> transposition;
	double minLength = 0.0;
	for (int i = 0; i < vertexes_.size(); i++)
	{
		transposition.push_back(i);
		int prev = (vertexes_.size() - 1 + i) % vertexes_.size();
		minLength += GetDistance(vertexes_[i], vertexes_[prev]);
	}

	while (std::next_permutation(transposition.begin(), transposition.end()))
	{
		double curLength = 0.0;
		for (int i = 0; i < transposition.size(); i++)
		{
			int prev = (transposition.size() - 1 + i) % transposition.size();
			curLength += GetDistance(vertexes_[transposition[i]], vertexes_[transposition[prev]]);
		}
		if (minLength > curLength)
			minLength = curLength;
	}
	return minLength;
}