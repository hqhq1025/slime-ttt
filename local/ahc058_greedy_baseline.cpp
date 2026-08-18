// Minimal reproducible AHC058 smoke baseline.
//
// It considers only level-0 upgrades. At each turn it chooses the affordable
// upgrade with the largest estimated remaining production / cost ratio, when
// that ratio exceeds one. The program maintains the exact official state so
// all emitted actions remain affordable.

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, levels, turns;
    long long apples;
    if (!(cin >> n >> levels >> turns >> apples)) return 1;

    vector<long long> production(n);
    for (auto& x : production) cin >> x;
    vector<vector<long long>> cost(levels, vector<long long>(n));
    vector<vector<long long>> count(levels, vector<long long>(n, 1));
    vector<vector<long long>> power(levels, vector<long long>(n, 0));
    for (auto& row : cost) {
        for (auto& x : row) cin >> x;
    }

    for (int turn = 0; turn < turns; ++turn) {
        int best_id = -1;
        long double best_ratio = 1.0L;
        const int remaining = turns - turn;
        for (int id = 0; id < n; ++id) {
            const long long price = cost[0][id] * (power[0][id] + 1);
            if (price > apples) continue;
            const long double ratio =
                static_cast<long double>(production[id]) * count[0][id] * remaining / price;
            if (ratio > best_ratio) {
                best_ratio = ratio;
                best_id = id;
            }
        }

        if (best_id >= 0) {
            cout << "0 " << best_id << '\n';
            apples -= cost[0][best_id] * (power[0][best_id] + 1);
            ++power[0][best_id];
        } else {
            cout << -1 << '\n';
        }

        for (int level = 0; level < levels; ++level) {
            for (int id = 0; id < n; ++id) {
                if (level == 0) {
                    apples += production[id] * count[level][id] * power[level][id];
                } else {
                    count[level - 1][id] += count[level][id] * power[level][id];
                }
            }
        }
    }
}
