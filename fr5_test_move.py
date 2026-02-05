from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.isaac.vscode")

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UrdfFileCfg


@configclass
class FR5SceneCfg(InteractiveSceneCfg):
    """시뮬레이션 장면 구성 설정 (설계도)"""
    
    # 1. 바닥과 조명 설정
    ground = AssetBaseCfg(
        prim_path="/Visuals/GroundPlane", 
        spawn=sim_utils.GroundPlaneCfg(
            #  아래 주소는 엔비디아에서 제공하는 공식 클라우드 자산 경로입니다.
            usd_path="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Environments/Grid/default_environment.usd"
        )
    )
    light = AssetBaseCfg(
        prim_path="/Visuals/Light", 
        spawn=sim_utils.DistantLightCfg(intensity=3000.0)
    )
    
    # 2. 로봇 소환 
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # [중요] GUI에서 'Save As'로 저장한 실제 경로를 넣으세요.
            usd_path="C:\\Users\\MSH\\Desktop\\lab\\robot_arm\\fr5.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
        ),
        # 관절 구동 방식 설정
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"], # 모든 관절에 적용
                stiffness=800.0,        # GUI에서 테스트했던 그 강성 값
                damping=40.0,           # GUI에서 테스트했던 그 감쇠 값
            ),
        },
    )



def main():
    # 1. 물리 시뮬레이션 환경 초기화
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    scene_cfg = FR5SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 로봇 객체 가져오기 (설계도에서 정의한 'robot' 이름으로 접근)
    robot = scene["robot"]
    
    sim.reset()
    print("로봇 관절 제어를 시작합니다. 화면을 확인하세요!")

    # 시뮬레이션 시간 추적을 위한 변수
    sim_time = 0.0


   # 2. 메인 루프
    while simulation_app.is_running():
        # 시간 경과에 따른 목표 각도 계산 (사인파)
        # targets의 shape은 (num_envs, num_joints)여야 합니다. 
        # 현재 num_envs=1 이므로 [[ ... ]] 형태의 2차원 텐서를 만듭니다.
        targets = torch.sin(torch.tensor([[sim_time] * robot.num_joints], device=sim.device)) * 0.5
        
        # [수정] 에러 메시지의 제안대로 's'를 빼고 호출합니다.
        robot.set_joint_position_target(targets)
        
        # 물리 엔진에 데이터 전달 및 한 단계 진행
        scene.write_data_to_sim()
        sim.step()
        
        # 시뮬레이션 결과 업데이트
        scene.update(dt=0.01)
        
        # 시간 업데이트
        sim_time += 0.01

    simulation_app.close()

if __name__ == "__main__":
    main()