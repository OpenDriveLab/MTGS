export CODEBASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $CODEBASE;
conda activate mtgs;
export PYTHONPATH=\"./\":$PYTHONPATH;

function mtgs_dev_setup(){
    tmp_var=$1;
    tmp_var=${tmp_var%.py}
    tmp_var=${tmp_var//\//.}
    export NERFSTUDIO_DATAPARSER_CONFIGS='nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser';
    export NERFSTUDIO_METHOD_CONFIGS='mtgs-dev='${tmp_var}':method';
    echo '-------------------------------------';
    echo 'ENV UPDATED with following ENV VARS.';
    echo '    NERFSTUDIO_DATAPARSER_CONFIGS = '${NERFSTUDIO_DATAPARSER_CONFIGS};
    echo '    NERFSTUDIO_METHOD_CONFIGS = '${NERFSTUDIO_METHOD_CONFIGS};
    echo '-------------------------------------'
}
echo 'INIT SETUP DONE.'

alias mtgs_setup="mtgs_dev_setup"
alias mtgs_train="ns-train mtgs-dev"
alias mtgs_render="python mtgs/tools/render.py interpolate --load-config"
alias mtgs_viewer="python mtgs/tools/run_viewer.py --viewer.camera-frustum-scale 0.3 --viewer.default_composite_depth False --viewer.max_num_display_images 500 --load-config"
