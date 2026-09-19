"""依赖优先的并行安装计划（POST /install-plan）。

调用本模块前，请求的 ``current_lock`` 已由 :func:`app.audit.audit_lock`
判定为**有效**（无 missing / unavailable / constraint_mismatch /
extraneous）。因此这里不做任何版本选择与约束核对，只在
:func:`app.paths.build_locked_graph` 构造的**根可达已激活依赖图**上做
结构分析（边 ``pkg -> dep`` 表示 pkg 依赖 dep，安装方向是 dep 先于 pkg）：

1. **强连通分量（SCC）**：用迭代版 Kosaraju（两遍 DFS，均不递归，环再大
   也不会栈溢出）把相互可达的包压成一个分量；
2. **缩点 DAG**：不同分量之间的依赖边去重，分量图天然无环；
3. **依赖优先分层**：在缩点 DAG 上取“到汇点的最长路径长度”为分量层级
   （被依赖最多的底层分量层级为 0），同一层的分量互不依赖、可并行安装；
   层级即批次下标——任一依赖所在批次严格早于依赖它的包，根包所在分量
   （唯一的源，其层级最大）恒在最后一批；
4. **同一循环分量作为一个 batch item**：``cyclic: true`` 并按包名序列出
   内部全部包（分量内的包必须作为一个单元一起安装）；自环（包依赖自身）
   也按单包循环分量标记；非循环 item 只含一个包。

确定性排序：分量内部包名、同一批次内的 item（先非循环后循环，再按
(首包名, 包名元组) 排序）、批次（层级）全部与输入顺序无关，结果只取决于
快照、目标环境与锁定清单。
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable


def _component_sort_key(members: Iterable[str]) -> tuple[str, ...]:
    """分量的确定性排序键：内部包名升序元组。"""
    return tuple(sorted(members))


def _strongly_connected_components(
    graph: dict[str, list[str]],
) -> list[frozenset[str]]:
    """迭代版 Kosaraju：返回强连通分量列表（每分量为包名 frozenset）。

    两遍 DFS 均显式管理栈、不递归：
    * 第一遍按字典序升序取未访问节点、沿已排序邻接边推进，按完成顺序入列；
    * 第二遍在逆图上按完成顺序的逆序取根（同样按字典序展开），每次 DFS
      到达的节点集即一个 SCC。
    遍历时节点与邻接均按包名升序，分量集合本身与遍历起点无关。
    """
    nodes = sorted(graph)

    # 逆图：与正图同步构造，邻接保持升序（graph[name] 已排序）。
    reverse: dict[str, list[str]] = {name: [] for name in nodes}
    for name in nodes:
        for dep in graph[name]:
            reverse[dep].append(name)

    # ---- 第一遍：完成顺序（finish 列表为完成先后，先完成者排最前）---- #
    finished: list[str] = []
    visited: set[str] = set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        # (节点, 下一条待检查邻接边的下标)；节点首次入栈即 visited。
        stack: list[tuple[str, int]] = [(start, 0)]
        while stack:
            name, edge_index = stack[-1]
            neighbours = graph[name]
            if edge_index < len(neighbours):
                nxt = neighbours[edge_index]
                stack[-1] = (name, edge_index + 1)
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append((nxt, 0))
            else:
                finished.append(name)
                stack.pop()

    # ---- 第二遍：逆图上按完成顺序逆序取根，逐棵 DFS 树即一个 SCC ---- #
    assigned: set[str] = set()
    components: list[frozenset[str]] = []
    for start in reversed(finished):
        if start in assigned:
            continue
        members: list[str] = []
        assigned.add(start)
        stack_rev: list[str] = [start]
        while stack_rev:
            name = stack_rev.pop()
            members.append(name)
            for nxt in reverse[name]:
                if nxt not in assigned:
                    assigned.add(nxt)
                    stack_rev.append(nxt)
        components.append(frozenset(members))

    return components


def build_install_batches(graph: dict[str, list[str]], *, root_name: str) -> dict:
    """把根可达已激活依赖图压成 SCC-DAG，输出依赖优先的并行安装批次。

    入参 ``graph`` 为 :func:`app.paths.build_locked_graph` 的结果
    （邻接表已按包名排序、去重）。返回
    ``{"batches": [{"index", "items": [...]}], "package_count": n}``：

    * 批次下标从 0 起按依赖方向递增，第 0 批是无任何图内依赖的最底层包；
    * 非循环 item：``{"packages": [name], "cyclic": false}``；
    * 循环 item：``{"packages": [内部包名...], "cyclic": true}``，包名升序；
    * 同批 item 先非循环后循环，再按 (packages[0], packages 元组) 升序。
    """
    components = _strongly_connected_components(graph)

    # 包 -> 所属分量（分量 frozenset 自身即确定性身份标识）。
    component_of: dict[str, frozenset[str]] = {}
    for members in components:
        for name in members:
            component_of[name] = members

    # 缩点 DAG：仅保留跨分量边并去重。边方向与原图一致 owner -> dep。
    condensed: dict[frozenset[str], set[frozenset[str]]] = {
        members: set() for members in components
    }
    for owner, deps in graph.items():
        owner_component = component_of[owner]
        for dep in deps:
            dep_component = component_of[dep]
            if dep_component != owner_component:
                condensed[owner_component].add(dep_component)

    # 层级 = 到汇点（最底层依赖）的最长路径长度：无图内依赖的分量层级为 0。
    # 依赖（被指向的分量）必须在更靠下的批次：先在缩点 DAG 上做 Kahn
    # 拓扑（汇点先出队），出队时其全部依赖层级已定，按 1 + max(依赖层级)
    # 取最长路径；就绪堆按分量排序键取最小，处理顺序不影响层级、仅为确定性。
    dependents: dict[frozenset[str], set[frozenset[str]]] = {
        members: set() for members in components
    }
    for owner_component, dep_components in condensed.items():
        for dep_component in dep_components:
            dependents[dep_component].add(owner_component)

    remaining_deps = {members: set(deps) for members, deps in condensed.items()}
    ready: list[tuple[tuple[str, ...], frozenset[str]]] = [
        (_component_sort_key(members), members)
        for members in components
        if not remaining_deps[members]
    ]
    heapq.heapify(ready)

    level: dict[frozenset[str], int] = {}
    processed = 0
    while ready:
        _, members = heapq.heappop(ready)
        dep_components = condensed[members]
        level[members] = (
            0 if not dep_components else 1 + max(level[dep] for dep in dep_components)
        )
        processed += 1
        for dependent in dependents[members]:
            remaining_deps[dependent].discard(members)
            if not remaining_deps[dependent]:
                heapq.heappush(
                    ready, (_component_sort_key(dependent), dependent)
                )

    # 全部为根可达节点，缩点 DAG 中每个分量都应被拓扑序覆盖。
    assert processed == len(components), "缩点 DAG 必须无环且全连通可达"

    def to_item(members: frozenset[str]) -> dict:
        packages = sorted(members)
        # 多包 SCC 恒为循环；单包 SCC 只有该包显式依赖自身（自环）才循环。
        cyclic = len(packages) > 1 or packages[0] in graph.get(packages[0], ())
        return {"packages": packages, "cyclic": cyclic}

    def item_sort_key(item: dict) -> tuple:
        # 同批内：非循环在前、循环在后，再按首包名与整个包名元组兜底。
        return (item["cyclic"], item["packages"][0], tuple(item["packages"]))

    root_component = component_of[root_name]
    root_level = level[root_component]

    batches: list[dict] = []
    for level_index in range(root_level + 1):
        items = [
            to_item(members)
            for members in components
            if level[members] == level_index
        ]
        items.sort(key=item_sort_key)
        batches.append({"index": level_index, "items": items})

    # 不变量：根是 SCC-DAG 唯一的源（所有包都从根可达），其最长路径层级
    # 最大，故根分量必然单独构成最后一批。
    assert batches[-1]["items"] == [to_item(root_component)]

    return {"batches": batches, "package_count": len(graph)}
