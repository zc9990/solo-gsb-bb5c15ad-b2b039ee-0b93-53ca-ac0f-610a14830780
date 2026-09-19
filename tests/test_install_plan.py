"""POST /install-plan 的端到端测试。

覆盖：先审计后规划的 409 短路（issues 与 /audit-lock 逐字节一致）、
线性链/单包/菱形的分层、多包 SCC 与自环作为一个 cyclic batch item、
SCC 内部及跨层依赖的批次先后不变量、根分量恒在最后一批、批内与包名
确定性排序（与仓库/锁键顺序无关）、marker 激活，以及 422 与额外字段校验。
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from packaging.utils import canonicalize_name

from app.main import app

client = TestClient(app)


def norm(name: str) -> str:
    return canonicalize_name(name)

ENV = {
    "python_version": "3.11",
    "sys_platform": "linux",
    "platform_machine": "x86_64",
}
WIN_ENV = {
    "python_version": "3.11",
    "sys_platform": "win32",
    "platform_machine": "AMD64",
}


def c(version, deps=None):
    return {"version": version, "dependencies": deps or []}


def d(name, spec="", marker=""):
    return {"name": name, "specifier": spec, "marker": marker}


def plan(repo, lock, *, root="app", env=None, **overrides):
    body = {
        "repository": repo,
        "root": root,
        "target_environment": env or ENV,
        "current_lock": lock,
    }
    body.update(overrides)
    return client.post("/install-plan", json=body)


def ok(repo, lock, **kw):
    r = plan(repo, lock, **kw)
    assert r.status_code == 200, r.text
    return r.json()


def batch_names(body):
    """batches -> [[包名...], ...]（展开 cyclic item）。"""
    return [
        [pkg for item in batch["items"] for pkg in item["packages"]]
        for batch in body["batches"]
    ]


def package_batch(body):
    """包名 -> 所在批次下标。"""
    return {
        pkg: batch["index"]
        for batch in body["batches"]
        for item in batch["items"]
        for pkg in item["packages"]
    }


def assert_install_invariants(body, graph, *, root="app"):
    """与任务要求直接对应的结构性不变量。"""
    batches = body["batches"]
    # 批次下标连续、从 0 递增。
    assert [b["index"] for b in batches] == list(range(len(batches)))
    # 任一依赖所在批次不得晚于（实际严格早于）依赖它的包；同批只能是 SCC 内部。
    where = package_batch(body)
    for owner, deps in graph.items():
        for dep in deps:
            if where[owner] != where[dep]:
                assert where[dep] < where[owner], (owner, dep, where)
    # 根包所在分量处于最后一批，且是最后一批唯一 item。
    last = batches[-1]["items"]
    assert len(last) == 1
    assert root in last[0]["packages"]
    # 每个包恰出现一次；package_count 与之吻合。
    all_packages = [pkg for names in batch_names(body) for pkg in names]
    assert len(all_packages) == len(set(all_packages))
    assert body["package_count"] == len(all_packages)
    # 批内 item 排序：非循环在前、循环在后，再按 (首包名, 包名元组)。
    for batch in batches:
        keys = [
            (item["cyclic"], item["packages"][0], tuple(item["packages"]))
            for item in batch["items"]
        ]
        assert keys == sorted(keys)
        for item in batch["items"]:
            assert item["packages"] == sorted(item["packages"])


def graph_of(repo, lock, env=ENV, root="app"):
    """按锁定版本展开已激活依赖，镜像服务端建图，供不变量核对。"""
    canon_repo = {norm(name): cands for name, cands in repo.items()}
    canon_lock = {norm(name): ver for name, ver in lock.items()}
    root = norm(root)
    graph: dict[str, list[str]] = {}
    seen = {root}
    queue = [root]
    while queue:
        name = queue.pop(0)
        deps = sorted({
            norm(x["name"])
            for cand in canon_repo[name]
            if cand["version"] == canon_lock[name]
            for x in cand.get("dependencies", [])
        })
        graph[name] = deps
        for dep in deps:
            if dep not in seen:
                seen.add(dep)
                queue.append(dep)
    return graph


# --------------------------------------------------------------------- #
# 1. 基础结构：单包、线性链、菱形并行
# --------------------------------------------------------------------- #
def test_single_root_package():
    repo = {"app": [c("1.0")]}
    body = ok(repo, {"app": "1.0"})
    assert body["batches"] == [
        {"index": 0, "items": [{"packages": ["app"], "cyclic": False}]}
    ]
    assert body["package_count"] == 1
    assert_install_invariants(body, {"app": []})


def test_linear_chain_batches():
    repo = {
        "app": [c("1.0", [d("liba")])],
        "liba": [c("1.0", [d("libb")])],
        "libb": [c("1.0", [d("libc")])],
        "libc": [c("1.0")],
    }
    lock = {"app": "1.0", "liba": "1.0", "libb": "1.0", "libc": "1.0"}
    body = ok(repo, lock)
    assert batch_names(body) == [["libc"], ["libb"], ["liba"], ["app"]]
    assert_install_invariants(body, graph_of(repo, lock))


def test_diamond_shares_a_batch_and_waits_for_dependencies():
    # app -> (a, b) -> leaf：leaf 第 0 批，a/b 第 1 批并行，app 最后。
    repo = {
        "app": [c("1.0", [d("a"), d("b")])],
        "a": [c("1.0", [d("leaf")])],
        "b": [c("1.0", [d("leaf")])],
        "leaf": [c("1.0")],
    }
    lock = {"app": "1.0", "a": "1.0", "b": "1.0", "leaf": "1.0"}
    body = ok(repo, lock)
    assert batch_names(body) == [["leaf"], ["a", "b"], ["app"]]
    assert_install_invariants(body, graph_of(repo, lock))


def test_independent_dependencies_pack_into_same_batch():
    # app 直接依赖 x/y/z 三个无内部依赖的包：全部第 0 批并行。
    repo = {
        "app": [c("1.0", [d("x"), d("y"), d("z")])],
        "x": [c("1.0")],
        "y": [c("1.0")],
        "z": [c("1.0")],
    }
    lock = {"app": "1.0", "x": "1.0", "y": "1.0", "z": "1.0"}
    body = ok(repo, lock)
    assert batch_names(body) == [["x", "y", "z"], ["app"]]


def test_levels_use_longest_path_dependency_is_never_later():
    # x: app->deep->mid->base（3 跳）；y: app->base 直连（1 跳）。
    # base 必须等到第 0 批；mid=1，deep=2，y=1，app=3。
    repo = {
        "app": [c("1.0", [d("deep"), d("y")])],
        "deep": [c("1.0", [d("mid")])],
        "mid": [c("1.0", [d("base")])],
        "y": [c("1.0", [d("base")])],
        "base": [c("1.0")],
    }
    lock = {"app": "1.0", "deep": "1.0", "mid": "1.0", "y": "1.0", "base": "1.0"}
    body = ok(repo, lock)
    assert batch_names(body) == [["base"], ["mid", "y"], ["deep"], ["app"]]
    assert_install_invariants(body, graph_of(repo, lock))


# --------------------------------------------------------------------- #
# 2. 循环：多包 SCC 与自环
# --------------------------------------------------------------------- #
def test_two_package_cycle_is_one_cyclic_item():
    repo = {
        "app": [c("1.0", [d("cyc-a")])],
        "cyc-a": [c("1.0", [d("cyc-b")])],
        "cyc-b": [c("1.0", [d("cyc-a")])],
    }
    lock = {"app": "1.0", "cyc-a": "1.0", "cyc-b": "1.0"}
    body = ok(repo, lock)
    assert body["batches"] == [
        {"index": 0,
         "items": [{"packages": ["cyc-a", "cyc-b"], "cyclic": True}]},
        {"index": 1,
         "items": [{"packages": ["app"], "cyclic": False}]},
    ]
    assert_install_invariants(body, graph_of(repo, lock))


def test_self_loop_is_cyclic_single_package_item():
    repo = {
        "app": [c("1.0", [d("selfy"), d("plain")])],
        "selfy": [c("1.0", [d("selfy")])],
        "plain": [c("1.0")],
    }
    lock = {"app": "1.0", "selfy": "1.0", "plain": "1.0"}
    body = ok(repo, lock)
    items0 = body["batches"][0]["items"]
    # 非循环 plain 排在 cyclic selfy 之前。
    assert items0 == [
        {"packages": ["plain"], "cyclic": False},
        {"packages": ["selfy"], "cyclic": True},
    ]
    assert_install_invariants(body, graph_of(repo, lock))


def test_cycle_with_external_dependency_waits_for_that_dependency():
    # cyc-a/cyc-b 构成 SCC，cyc-b 还依赖 base：SCC 必须晚于 base。
    repo = {
        "app": [c("1.0", [d("cyc-a")])],
        "cyc-a": [c("1.0", [d("cyc-b")])],
        "cyc-b": [c("1.0", [d("cyc-a"), d("base")])],
        "base": [c("1.0")],
    }
    lock = {"app": "1.0", "cyc-a": "1.0", "cyc-b": "1.0", "base": "1.0"}
    body = ok(repo, lock)
    assert batch_names(body) == [["base"], ["cyc-a", "cyc-b"], ["app"]]
    cyclic_item = body["batches"][1]["items"][0]
    assert cyclic_item["cyclic"] is True
    assert_install_invariants(body, graph_of(repo, lock))


def test_larger_scc_remains_one_item_at_correct_level():
    # 三节点 SCC {t1,t2,t3}，外部依赖 base，被 app 依赖。
    repo = {
        "app": [c("1.0", [d("t1")])],
        "t1": [c("1.0", [d("t2")])],
        "t2": [c("1.0", [d("t3")])],
        "t3": [c("1.0", [d("t1"), d("base")])],
        "base": [c("1.0")],
    }
    lock = {"app": "1.0", "t1": "1.0", "t2": "1.0", "t3": "1.0", "base": "1.0"}
    body = ok(repo, lock)
    assert batch_names(body) == [["base"], ["t1", "t2", "t3"], ["app"]]
    assert body["batches"][1]["items"] == [
        {"packages": ["t1", "t2", "t3"], "cyclic": True}
    ]
    assert_install_invariants(body, graph_of(repo, lock))


def test_non_cyclic_single_package_item_is_never_marked_cyclic():
    # 两个包互相可达才是 SCC；菱形汇合不产生 cyclic 标记。
    repo = {
        "app": [c("1.0", [d("a"), d("b")])],
        "a": [c("1.0", [d("leaf")])],
        "b": [c("1.0", [d("leaf")])],
        "leaf": [c("1.0")],
    }
    lock = {"app": "1.0", "a": "1.0", "b": "1.0", "leaf": "1.0"}
    body = ok(repo, lock)
    assert all(
        item["cyclic"] is False
        for batch in body["batches"]
        for item in batch["items"]
    )


# --------------------------------------------------------------------- #
# 3. 确定性：与仓库/锁键/候选声明顺序无关
# --------------------------------------------------------------------- #
def test_deterministic_regardless_of_input_order():
    import random

    repo = {
        "app": [c("3.0", [d("cyc-a"), d("core"), d("selfy")]), c("2.0")],
        "core": [c("2.0", [d("util"), d("base")]), c("1.0")],
        "util": [c("1.0", [d("base")])],
        "base": [c("1.0")],
        "cyc-a": [c("1.0", [d("cyc-b")])],
        "cyc-b": [c("1.0", [d("cyc-a")])],
        "selfy": [c("1.0", [d("selfy")])],
    }
    lock = {
        "app": "3.0", "core": "2.0", "util": "1.0", "base": "1.0",
        "cyc-a": "1.0", "cyc-b": "1.0", "selfy": "1.0",
    }
    baseline = ok(repo, lock)

    def shuffled(value):
        rng = random.Random(1234)
        if isinstance(value, dict):
            items = list(value.items())
            rng.shuffle(items)
            return {k: shuffled(v) for k, v in items}
        if isinstance(value, list):
            items = value[:]
            rng.shuffle(items)
            return [shuffled(v) for v in items]
        return value

    for _ in range(8):
        assert ok(shuffled(repo), shuffled(lock)) == baseline


def test_pep503_name_normalisation_in_plan():
    # 锁键与快照名大小写/下划线/点不同形：规范化后同一计划。
    repo = {
        "App": [c("1.0", [d("Some_Lib")])],
        "some.lib": [c("1.0")],
    }
    lock = {"app": "1.0", "some-lib": "1.0"}
    body = ok(repo, lock, root="App")
    assert batch_names(body) == [["some-lib"], ["app"]]
    assert body["root"] == "app"


# --------------------------------------------------------------------- #
# 4. marker：只对已激活依赖建图
# --------------------------------------------------------------------- #
def test_inactive_marker_dependencies_excluded_from_plan():
    repo = {
        "app": [
            c("1.0", [
                d("always"),
                d("win-only", marker="sys_platform == 'win32'"),
            ])
        ],
        "always": [c("1.0")],
        "win-only": [c("1.0")],
    }
    lock = {"app": "1.0", "always": "1.0", "win-only": "1.0"}
    # linux 下 win-only 不可达 -> extraneous，锁无效，409。
    r = plan(repo, lock, env=ENV)
    assert r.status_code == 409
    issue = r.json()["issues"][0]
    assert issue["type"] == "extraneous" and issue["name"] == "win-only"

    # win32 下两者均激活，同一锁有效。
    body = ok(repo, lock, env=WIN_ENV)
    assert batch_names(body) == [["always", "win-only"], ["app"]]


# --------------------------------------------------------------------- #
# 5. 审计短路与请求校验
# --------------------------------------------------------------------- #
def test_invalid_lock_returns_409_with_audit_issues_and_no_batches():
    repo = {
        "app": [c("1.0", [d("present"), d("missing-lib")])],
        "present": [c("1.0")],
        "missing-lib": [c("1.0")],
    }
    lock = {"app": "1.0", "present": "1.0", "ghost": "9.9"}
    r = plan(repo, lock)
    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "invalid_lock"
    assert "batches" not in body
    types = {issue["type"] for issue in body["issues"]}
    assert types == {"missing", "extraneous"}

    # 与直接调 /audit-lock 的 issues 逐字节一致。
    audit = client.post(
        "/audit-lock",
        json={"repository": repo, "root": "app", "target_environment": ENV,
              "current_lock": lock},
    ).json()
    assert body["issues"] == audit["issues"]


def test_constraint_mismatch_lock_returns_409():
    repo = {
        "app": [c("1.0", [d("lib", ">=2.0")])],
        "lib": [c("2.0"), c("1.0")],
    }
    r = plan(repo, {"app": "1.0", "lib": "1.0"})
    assert r.status_code == 409
    assert r.json()["issues"][0]["type"] == "constraint_mismatch"


def test_unknown_root_returns_422():
    r = plan({"app": [c("1.0")]}, {"app": "1.0"}, root="missing")
    assert r.status_code == 422
    assert r.json()["status"] == "invalid_snapshot"


def test_bad_lock_version_returns_422():
    repo = {"app": [c("1.0", [d("lib")])], "lib": [c("1.0")]}
    r = plan(repo, {"app": "1.0", "lib": "oops"})
    assert r.status_code == 422
    assert "PEP 440" in r.json()["message"]


def test_extra_field_rejected():
    r = client.post(
        "/install-plan",
        json={
            "repository": {"app": [c("1.0")]},
            "root": "app",
            "target_environment": ENV,
            "current_lock": {"app": "1.0"},
            "unexpected": 1,
        },
    )
    assert r.status_code == 422


def test_openapi_documents_responses():
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/install-plan"]["post"]
    assert set(op["responses"]) == {"200", "409", "422"}
