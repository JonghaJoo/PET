import numpy as np
import openseespy.opensees as op

### Run Opensees to obtain the damping matrix and the target response


def dynamic_2DOF(mat_type, mat_props, t, gm, gm_scale=1, beta_k_set=None):
    op.wipe()
    ndm = 2
    ndf = 3
    op.model('basic', '-ndm', ndm, '-ndf', ndf)
    bot_node = 1
    top_node = 3
    op.node(bot_node, 0., 0.)
    op.node(top_node, 0., 0.)
    mid_node = 2
    op.node(mid_node, 0., 0.)
    op.fix(bot_node, *[1, 1, 1])
    op.fix(top_node, *[0, 1, 1])
    op.fix(mid_node, *[0, 1, 1])
    op.equalDOF(bot_node, top_node, *[2, 3])
    op.equalDOF(bot_node, mid_node, *[2, 3])
    op.mass(top_node, 1.0, 0., 0.)
    op.mass(mid_node, 1.0, 0., 0.)
    mat_tag = 1
    op.uniaxialMaterial(mat_type, 1, *mat_props)
    beam_mid_tag = 1
    beam_top_tag = 2
    op.element('zeroLength', beam_mid_tag, bot_node, mid_node, '-mat', mat_tag,
              '-dir', 1, '-doRayleigh', 1)
    op.element('zeroLength', beam_top_tag, mid_node, top_node, '-mat', mat_tag,
              '-dir', 1, '-doRayleigh', 1)
    load_tag_dynamic = 1
    pattern_tag_dynamic = 1
    dt = t[1]-t[0]
    values = (-gm*9.8*gm_scale).tolist()
    op.timeSeries('Path', load_tag_dynamic, '-dt', dt, '-values', *values)
    op.pattern('UniformExcitation', pattern_tag_dynamic, 1, '-accel', load_tag_dynamic)
    xi = 0.00
    angular_freq = ((op.eigen('-fullGenLapack', 1))[0])**0.5
    alpha_m = 0.0
    beta_kk = 2*xi/angular_freq
    beta_k = 0.0
    beta_k_comm = 0.0
    beta_k_init = 0.0
    if beta_k_set == None:
        beta_k = beta_kk
    elif beta_k_set == 'init':
        beta_k_init = beta_kk

    op.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)
    op.wipeAnalysis()
    op.algorithm('Newton')
    op.system('SparseGeneral')
    op.numberer('RCM')
    op.constraints('Transformation')
    op.integrator('Newmark', 0.5, 0.25)
    op.analysis('Transient')
    tol = 1.0e-10
    iterations = 10
    op.test('EnergyIncr', tol, iterations, 0, 2)
    analysis_time = t[-1]
    analysis_dt = dt
    outputs = {
        'time': [],
        'disp_top' : [],
        'accel_top' : [],
        'vel_top' : [],
        'disp_mid' : [],
        'accel_mid' : [],
        'vel_mid' : [],
        'force_1' : [],
        'force_2' : []
    }
    while op.getTime() < analysis_time:
        curr_time = op.getTime()
        op.analyze(1, analysis_dt)
        outputs['time'].append(curr_time)
        outputs['disp_top'].append(op.nodeDisp(top_node, 1))
        outputs['vel_top'].append(op.nodeVel(top_node, 1))
        outputs['accel_top'].append(op.nodeAccel(top_node, 1))
        outputs['disp_mid'].append(op.nodeDisp(mid_node, 1))
        outputs['vel_mid'].append(op.nodeVel(mid_node, 1))
        outputs['accel_mid'].append(op.nodeAccel(mid_node, 1))
        outputs['force_1'].append(op.eleForce(beam_mid_tag, 1))
        outputs['force_2'].append(op.eleForce(beam_top_tag, 1))
    op.wipe()
    for item in outputs:
        outputs[item] = np.array(outputs[item])
    return outputs, beta_kk


def dynamic_1DOF(mat_type, mat_props, t, gm, gm_scale=1, beta_k_set=None):
    op.wipe()

    ndm = 2
    ndf = 3
    op.model('basic', '-ndm', ndm, '-ndf', ndf)

    bot_node = 1
    top_node = 2
    op.node(bot_node, 0., 0.)
    op.node(top_node, 0., 0.)

    op.fix(bot_node, *[1, 1, 1])
    op.fix(top_node, *[0, 1, 1])

    op.equalDOF(bot_node, top_node, *[2, 3])

    op.mass(top_node, 1.0, 0., 0.)

    mat_tag = 1

    op.uniaxialMaterial(mat_type, 1, *mat_props)

    beam_tag = 1
    op.element('zeroLength', beam_tag, bot_node, top_node, '-mat', mat_tag,
              '-dir', 1, '-doRayleigh', 1)

    load_tag_dynamic = 1
    pattern_tag_dynamic = 1

    dt = t[1]-t[0]

    values = (-9.8*gm*gm_scale).tolist()

    op.timeSeries('Path', load_tag_dynamic, '-dt', dt, '-values', *values)
    op.pattern('UniformExcitation', pattern_tag_dynamic, 1, '-accel', load_tag_dynamic)

    xi = 0.00
    angular_freq = ((op.eigen('-fullGenLapack', 1))[0])**0.5
    alpha_m = 0.0
    beta_kk = 2*xi/angular_freq
    beta_k = 0.0
    beta_k_comm = 0.0
    beta_k_init = 0.0
    if beta_k_set == None:
        beta_k = beta_kk
    elif beta_k_set == 'init':
        beta_k_init = beta_kk

    op.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)

    op.wipeAnalysis()
    op.algorithm('Newton')
    op.system('SparseGeneral')
    op.numberer('RCM')
    op.constraints('Transformation')
    op.integrator('Newmark', 0.5, 0.25)
    op.analysis('Transient')

    tol = 1.0e-10
    iterations = 10
    op.test('EnergyIncr', tol, iterations, 0, 2)
    analysis_time = t[-1]
    analysis_dt = dt/2
    outputs = {
        'time': [],
        'rel_disp' : [],
        'rel_accel' : [],
        'rel_vel' : [],
        'force' : [],
        'force_2' : []
    }

    while op.getTime() < analysis_time:
        curr_time = op.getTime()
        op.analyze(1, analysis_dt)
        outputs['time'].append(curr_time)
        outputs['rel_disp'].append(op.nodeDisp(top_node, 1))
        outputs['rel_vel'].append(op.nodeVel(top_node, 1))
        outputs['rel_accel'].append(op.nodeAccel(top_node, 1))
        op.reactions()
        outputs['force'].append(-op.nodeReaction(bot_node, 1))
        outputs['force_2'].append(op.eleForce(beam_tag, 1))

    op.wipe()
    for item in outputs:
        outputs[item] = np.array(outputs[item])
    return outputs, beta_kk
    

def static_1DOF(mat_type, mat_props, disp):
    op.wipe()

    ndm = 2
    ndf = 3
    op.model('basic', '-ndm', ndm, '-ndf', ndf)

    bot_node = 1
    top_node = 2
    op.node(bot_node, 0., 0.)
    op.node(top_node, 0., 0.)

    op.fix(bot_node, *[1, 1, 1])
    op.fix(top_node, *[0, 1, 1])

    op.equalDOF(bot_node, top_node, *[2, 3])

    op.mass(top_node, 1.0, 0., 0.)

    mat_tag = 1

    op.uniaxialMaterial(mat_type, 1, *mat_props)

    beam_tag = 1
    op.element('zeroLength', beam_tag, bot_node, top_node, '-mat', mat_tag,
              '-dir', 1, '-doRayleigh', 1)

    pattern_tag = 1
    ts_tag = 1

    op.timeSeries('Constant',1)
    op.pattern('Plain', pattern_tag , ts_tag)
    op.load(top_node, 10., 0., 0.)
    disp_diff = np.diff(disp, prepend=0)
    disp_diff = disp_diff.tolist()
    disp = disp.tolist()


    op.wipeAnalysis()
    op.algorithm('Newton')
    op.system('SparseGeneral')
    op.numberer('RCM')
    op.constraints('Transformation')
    op.analysis('Static')

    tol = 1.0e-10
    iterations = 10
    op.test('EnergyIncr', tol, iterations, 0, 2)
    outputs = {
        'disp' : [],
        'force' : [],
    }

    for i in range(len(disp_diff)):
        op.integrator('DisplacementControl', top_node, 1, disp_diff[i])
        op.analyze(1)
        outputs['disp'].append(op.nodeDisp(top_node, 1))
        op.reactions()
        outputs['force'].append(-op.eleForce(beam_tag, 1))
        op.integrator('DisplacementControl', top_node, 1, -disp_diff[i])

    op.wipe()
    for item in outputs:
        outputs[item] = np.array(outputs[item])
    return outputs