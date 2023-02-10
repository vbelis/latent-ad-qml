from .compute_expr_ent import (
    main,
    prepare_circs,
    compute_expr_ent_vs_circuit,
    expr_vs_nqubits,
    var_kernel_vs_nqubits,
    get_data,
    u_dense_encoding_no_ent,
    u_dense_encoding,
    u_dense_encoding_all,
    get_arguments,
)

from .plot import (
    get_roc_data,
    get_FPR_for_fixed_TPR,
    get_mean_and_error,
    plot_ROC_kfold_mean,
    create_table_for_fixed_TPR,
    
)