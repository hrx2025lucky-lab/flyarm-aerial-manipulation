function varargout = flyarm_compare_pid_ladrc(varargin)
%FLYARM_COMPARE_PID_LADRC Run PID and LADRC under the same scenario.

if nargout == 0
    flyarm_compare_pd_ladrc(varargin{:});
else
    [varargout{1:nargout}] = flyarm_compare_pd_ladrc(varargin{:});
end
end
