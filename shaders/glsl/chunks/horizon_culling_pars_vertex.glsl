// reference: https://cesium.com/blog/2013/04/25/horizon-culling/
#include "ellipsoid.glsl"
/**
 * Determines if a point is hidden below the horizon from the camera.
 * The occluder is the ellipsoid shrunk by `nvrHorizonMinHeight` (never
 * positive), scaled down further to pass through the point or the camera when
 * either lies below it: a point is never hidden by the ellipsoid it is under,
 * and a camera under the ellipsoid culls at its own tangent plane.
 * Note: Does not account for the cone test.
 *
 * @name nvr_horizon_culled
 * @glslFunction
 *
 * param {vec3} targetPosition World position of the point in ECEF meters.
 * param {vec3} cameraPosition World position of the camera in the same space.
 * returns {bool} true if the point lies beyond the ellipsoidal horizon and can be culled.
 */
flat out int vHorizonCulled;
uniform float nvrHorizonMinHeight;

bool nvr_horizon_culled(vec3 targetPosition, vec3 cameraPosition) {
    vec3 oneOverRadii = 1.0 / (WGS84_RADII + nvrHorizonMinHeight);
    vec3 cameraPositionScaled = cameraPosition * oneOverRadii;
    vec3 targetPositionScaled = targetPosition * oneOverRadii;

    float occluderScaleSquared = min(
        1.0,
        min(dot(targetPositionScaled, targetPositionScaled),
            dot(cameraPositionScaled, cameraPositionScaled)));

    return dot(targetPositionScaled, cameraPositionScaled) < occluderScaleSquared;
}
