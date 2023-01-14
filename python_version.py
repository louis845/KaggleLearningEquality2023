import sys
import pkg_resources

print(sys.version)
print([p.project_name for p in pkg_resources.working_set])