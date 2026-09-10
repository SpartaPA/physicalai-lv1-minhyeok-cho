# `src/` 변경 사항 — Module 3

`06_chain.ipynb` 완성을 위해 수정한 `src/` 모듈의 변경 사항입니다.

## `src/coordinate_chain.py`

`CoordinateChain`을 부모 → 자식 동차변환을 조립하는 작은 TF2 형태의 유틸리티로 완성했습니다.

- `_path_to_root(frame)`
  - 지정 프레임에서 root까지의 경로를 반환합니다.
  - root에 연결되지 않은 프레임은 이해하기 쉬운 `KeyError`를 발생시킵니다.
  - 잘못 등록된 순환 체인은 `ValueError`로 감지합니다.
- `T_from_root(frame)`
  - 경로의 변환을 올바른 순서로 곱해 `T(root <- frame)`을 만듭니다.
- `T(target, source)`
  - `inv(T(root <- target)) @ T(root <- source)`로 임의 두 프레임 사이의 변환을 계산합니다.
- `transform(target, source, P, w)`
  - 기존 `transform_points`를 재사용해 `(3,)` 단일 점과 `(N, 3)` 점군을 반복문 없이 변환합니다.
  - `w=1`은 점(회전 + 병진), `w=0`은 방향(회전만)을 나타냅니다.
- `default_chain()`
  - 과제에서 지정한 회전각을 라디안으로 변환해 base → link → camera 체인을 구성합니다.
- `camera_point_to_base()` / `base_point_to_camera()`
  - 기본 체인 또는 전달된 체인을 사용해 양방향 좌표 변환을 수행합니다.

## `src/rotation.py`

- `quaternion_from_axis_angle()`의 반환 순서를 SciPy와 동일한 `(x, y, z, w)`로 수정했습니다.
  - 이전 구현은 스칼라 성분을 먼저 반환하는 `(w, x, y, z)` 순서였습니다.
  - 이 변경으로 `Rotation.as_quat()` 결과와 직접 비교할 수 있으며, 노트북 6-5의 쿼터니언 검증을 만족합니다.

## 검증

- `06_chain.ipynb`의 제공 검증 6-1 ~ 6-5를 순서대로 실행해 모두 통과했습니다.
- 100만 개 점군의 camera → base → camera 왕복 최대 오차는 약 `6.7e-16 m`였습니다.
- `coordinate_chain.py`의 단일 점·점군 왕복, 변환 합성, 축-각 및 쿼터니언 형식을 추가로 확인했습니다.

> 참고: `tests/test_transform.py`의 6개 실패는 이번 변경과 무관한 기존 `NotImplementedError` 플레이스홀더입니다.
