function o = sv_options(name)
% sv_options - the options of ssm.models.sv for each stochastic volatility model of the
% tutorial; every model keeps the default priors and a mean
switch name
    case 'SV',      o = {};
    case 'SV-t',    o = {'dist', 't'};
    case 'SV-RW',   o = {'vol', 'rw'};
    case 'SV-RW-t', o = {'vol', 'rw', 'dist', 't'};
    case 'SV-MA',   o = {'errors', 'ma'};
    otherwise, error('sv_options: unknown model ''%s''', name);
end
end
