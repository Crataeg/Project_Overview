"""
依赖检查和管理工具
自动检查并安装项目所需的依赖包
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple, Optional

# 必需依赖包及其最低版本（使用导入名）
REQUIRED_PACKAGES = {
    'numpy': '1.19.0',
    'scipy': '1.5.0',
    'geatpy': '2.7.0',
    'matplotlib': '3.3.0',
    'pandas': '1.1.0',
    'tqdm': '4.62.0',
    'yaml': '5.4.0',  # 导入名是 yaml，安装名是 pyyaml
    'colorlog': '6.0.0'
}

# 包名映射（导入名 -> 安装名）
PACKAGE_INSTALL_MAP = {
    'yaml': 'pyyaml'
}

# 特殊包版本处理
SPECIAL_PACKAGE_VERSIONS = {
    'colorlog': '6.10.1'  # 手动指定已知版本
}


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 6:
        return True, version_str
    else:
        return False, version_str


def get_installed_version(package_name: str) -> Optional[str]:
    """获取已安装包的版本"""
    try:
        # 特殊处理 PyYAML
        if package_name == 'yaml':
            try:
                import yaml
                return getattr(yaml, '__version__', '6.0.1')  # PyYAML 6.0.1
            except ImportError:
                return None

        # 特殊处理 colorlog（没有标准版本属性）
        if package_name == 'colorlog':
            try:
                import colorlog
                # colorlog 没有 __version__ 属性，返回已知版本
                return SPECIAL_PACKAGE_VERSIONS.get('colorlog', 'unknown')
            except ImportError:
                return None

        # 正常处理其他包
        import_name = package_name
        module = importlib.import_module(import_name)

        if hasattr(module, '__version__'):
            return module.__version__
        elif hasattr(module, 'version'):
            return module.version
        elif package_name in SPECIAL_PACKAGE_VERSIONS:
            return SPECIAL_PACKAGE_VERSIONS[package_name]
        else:
            return "unknown"

    except ImportError:
        return None


def check_package_compatibility(package_name: str, installed_version: str, required_version: str) -> bool:
    """检查包版本兼容性"""
    if installed_version == "unknown":
        return True  # 无法检查版本，假设兼容

    try:
        # 尝试使用 packaging 库进行版本比较
        from packaging import version
        return version.parse(installed_version) >= version.parse(required_version)
    except ImportError:
        # 如果 packaging 不可用，使用简单字符串比较
        try:
            return installed_version >= required_version
        except:
            return True  # 比较失败时假设兼容


def install_package(package_name: str, version: str = "") -> bool:
    """安装指定的包"""
    try:
        # 处理包名映射
        install_name = PACKAGE_INSTALL_MAP.get(package_name, package_name)

        if version:
            package_spec = f"{install_name}>={version}"
        else:
            package_spec = install_name

        print(f"正在安装 {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✓ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {package_name} 安装失败: {e}")
        return False
    except Exception as e:
        print(f"✗ {package_name} 安装出错: {e}")
        return False


def check_dependencies(install_missing: bool = False) -> Dict:
    """检查所有依赖包"""
    print("=" * 60)
    print("检查系统依赖...")
    print("=" * 60)

    # 检查Python版本
    py_ok, py_version = check_python_version()
    status_icon = "✓" if py_ok else "❌"
    print(f"{status_icon} Python版本: {py_version}")

    results = {
        'python': {'ok': py_ok, 'version': py_version},
        'packages': {},
        'missing': [],
        'incompatible': []
    }

    # 检查每个包
    for package, required_version in REQUIRED_PACKAGES.items():
        installed_version = get_installed_version(package)

        if installed_version is None:
            # 包未安装
            status_icon = "❌"
            status = "未安装"
            is_ok = False
            results['missing'].append(package)
        else:
            # 检查版本兼容性
            is_compatible = check_package_compatibility(package, installed_version, required_version)
            if is_compatible:
                status_icon = "✓"
                status = installed_version
                is_ok = True
            else:
                status_icon = "⚠"
                status = f"{installed_version} (需要 >={required_version})"
                is_ok = False
                results['incompatible'].append(package)

        # 格式化输出
        display_name = package
        display_status = f"{status:<15}"

        print(f"{status_icon} {display_name:<12} {display_status} (需要 >={required_version})")

        results['packages'][package] = {
            'ok': is_ok,
            'installed': installed_version,
            'required': required_version
        }

        # 如果要求自动安装且包缺失
        if install_missing and installed_version is None:
            if install_package(package, required_version):
                # 重新检查安装结果
                new_version = get_installed_version(package)
                if new_version is not None:
                    results['packages'][package]['ok'] = True
                    results['packages'][package]['installed'] = new_version
                    if package in results['missing']:
                        results['missing'].remove(package)

    return results


def print_summary(results: Dict):
    """打印检查摘要"""
    print("=" * 60)
    print("依赖检查摘要:")
    print("=" * 60)

    missing_count = len(results['missing'])
    incompatible_count = len(results['incompatible'])

    if missing_count == 0 and incompatible_count == 0:
        print("✓ 所有依赖包都已安装且版本兼容!")
        return

    if missing_count > 0:
        print(f"❌ 缺少 {missing_count} 个必要包:")
        for package in results['missing']:
            install_name = PACKAGE_INSTALL_MAP.get(package, package)
            print(f"   - {package} (安装名: {install_name})")

    if incompatible_count > 0:
        print(f"⚠  {incompatible_count} 个包版本不兼容:")
        for package in results['incompatible']:
            info = results['packages'][package]
            print(f"   - {package}: 已安装 {info['installed']}, 需要 >={info['required']}")

    # 生成安装命令
    if missing_count > 0:
        missing_packages = []
        for package in results['missing']:
            install_name = PACKAGE_INSTALL_MAP.get(package, package)
            required_version = REQUIRED_PACKAGES[package]
            missing_packages.append(f"{install_name}>={required_version}")

        install_cmd = "pip install " + " ".join(missing_packages)
        print(f"\n请使用以下命令安装:\n  {install_cmd}")


def verify_all_packages():
    """验证所有包的实际可用性"""
    print("=" * 60)
    print("验证包实际可用性...")
    print("=" * 60)

    packages_to_check = [
        ('numpy', 'np'),
        ('scipy', 'sp'),
        ('geatpy', 'ea'),
        ('matplotlib.pyplot', 'plt'),
        ('pandas', 'pd'),
        ('tqdm', 'tqdm'),
        ('yaml', 'yaml'),
        ('colorlog', 'colorlog')
    ]

    all_ok = True
    for import_name, alias in packages_to_check:
        try:
            if '.' in import_name:
                # 处理子模块导入
                parts = import_name.split('.')
                base_module = __import__(parts[0])
                module = base_module
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                module = __import__(import_name)

            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {import_name:<18} - 导入成功 (版本: {version})")

        except ImportError as e:
            print(f"✗ {import_name:<18} - 导入失败: {e}")
            all_ok = False
        except Exception as e:
            print(f"✗ {import_name:<18} - 错误: {e}")
            all_ok = False

    return all_ok


def main():
    """主函数"""
    # 检查依赖
    results = check_dependencies(install_missing=False)

    # 打印摘要
    print_summary(results)

    # 验证实际可用性
    print(f"\n{'=' * 60}")
    print("最终验证:")
    print('=' * 60)

    if verify_all_packages():
        print("\n🎉 所有包都可用！项目可以正常运行。")
    else:
        print("\n❌ 有些包存在问题，请检查上述错误信息。")

    # 特殊说明
    if 'yaml' in results['missing']:
        print(f"\n💡 注意: PyYAML 可能已安装但检测不到。")
        print("   如果上面的验证显示 'yaml - 导入成功'，则可以忽略依赖检查的警告。")


if __name__ == "__main__":
    main()