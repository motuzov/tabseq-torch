from dagster import (
    AssetsDefinition,
    Definitions,
    define_asset_job,
    file_relative_path,
)
from dagstermill import (
    ConfigurableLocalOutputNotebookIOManager,
    define_dagstermill_asset,
)


def _define_dagstermill_asset(
    file_name: str, deps: AssetsDefinition | None, local_path
) -> AssetsDefinition:
    name = file_name.split(".")[0]
    file_name = f"{local_path}/{file_name}" if local_path else file_name
    return define_dagstermill_asset(
        name=name,
        notebook_path=file_relative_path(__file__, file_name),
        deps=[deps] if deps else deps,
        save_notebook_on_failure=True,
    )


ASSETS: list[AssetsDefinition] = []


def _define_asset_job(nb_file_names: list[str], name: str, local_path=""):
    if not nb_file_names:
        raise (ValueError())
    asset_defs = [_define_dagstermill_asset(nb_file_names[0], None, local_path)]
    for nb_file_name in nb_file_names[1:]:
        asset_defs.append(
            _define_dagstermill_asset(nb_file_name, asset_defs[-1], local_path)
        )
    ASSETS.extend(asset_defs)
    return define_asset_job(name=name, selection=asset_defs)


fast_test_job = _define_asset_job(
    nb_file_names=["daily_agg.ipynb", "daily_win.ipynb"], name="fast_daily_tests_job"
)


unit_test_data_viz_job = _define_asset_job(
    nb_file_names=[
        "agg_pl.ipynb",
        "win_pl.ipynb",
    ],
    name="unit_test_data_viz",
    local_path="unit_test_data_viz",
)


daily_assets_job = _define_asset_job(
    nb_file_names=[
        "daily_assets.ipynb",
    ],
    name="daily_assets_job",
)


defs = Definitions(
    assets=ASSETS,
    jobs=[fast_test_job, unit_test_data_viz_job, daily_assets_job],
    resources={
        "output_notebook_io_manager": ConfigurableLocalOutputNotebookIOManager(),
    },
)
